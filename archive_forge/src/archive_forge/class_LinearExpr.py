import collections
import itertools
import numbers
import threading
import time
from typing import (
import warnings
import pandas as pd
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model_helper as cmh
from ortools.sat.python import swig_helper
from ortools.util.python import sorted_interval_list
class LinearExpr:
    """Holds an integer linear expression.

    A linear expression is built from integer constants and variables.
    For example, `x + 2 * (y - z + 1)`.

    Linear expressions are used in CP-SAT models in constraints and in the
    objective:

    * You can define linear constraints as in:

    ```
    model.add(x + 2 * y <= 5)
    model.add(sum(array_of_vars) == 5)
    ```

    * In CP-SAT, the objective is a linear expression:

    ```
    model.minimize(x + 2 * y + z)
    ```

    * For large arrays, using the LinearExpr class is faster that using the python
    `sum()` function. You can create constraints and the objective from lists of
    linear expressions or coefficients as follows:

    ```
    model.minimize(cp_model.LinearExpr.sum(expressions))
    model.add(cp_model.LinearExpr.weighted_sum(expressions, coefficients) >= 0)
    ```
    """

    @classmethod
    def sum(cls, expressions: Sequence[LinearExprT]) -> LinearExprT:
        """Creates the expression sum(expressions)."""
        if len(expressions) == 1:
            return expressions[0]
        return _SumArray(expressions)

    @overload
    @classmethod
    def weighted_sum(cls, expressions: Sequence[LinearExprT], coefficients: Sequence[IntegralT]) -> LinearExprT:
        ...

    @overload
    @classmethod
    def weighted_sum(cls, expressions: Sequence[ObjLinearExprT], coefficients: Sequence[NumberT]) -> ObjLinearExprT:
        ...

    @classmethod
    def weighted_sum(cls, expressions, coefficients):
        """Creates the expression sum(expressions[i] * coefficients[i])."""
        if LinearExpr.is_empty_or_all_null(coefficients):
            return 0
        elif len(expressions) == 1:
            return expressions[0] * coefficients[0]
        else:
            return _WeightedSum(expressions, coefficients)

    @overload
    @classmethod
    def term(cls, expressions: LinearExprT, coefficients: IntegralT) -> LinearExprT:
        ...

    @overload
    @classmethod
    def term(cls, expressions: ObjLinearExprT, coefficients: NumberT) -> ObjLinearExprT:
        ...

    @classmethod
    def term(cls, expression, coefficient):
        """Creates `expression * coefficient`."""
        if cmh.is_zero(coefficient):
            return 0
        else:
            return expression * coefficient

    @classmethod
    def is_empty_or_all_null(cls, coefficients: Sequence[NumberT]) -> bool:
        for c in coefficients:
            if not cmh.is_zero(c):
                return False
        return True

    @classmethod
    def rebuild_from_linear_expression_proto(cls, model: cp_model_pb2.CpModelProto, proto: cp_model_pb2.LinearExpressionProto) -> LinearExprT:
        """Recreate a LinearExpr from a LinearExpressionProto."""
        offset = proto.offset
        num_elements = len(proto.vars)
        if num_elements == 0:
            return offset
        elif num_elements == 1:
            return IntVar(model, proto.vars[0], None) * proto.coeffs[0] + offset
        else:
            variables = []
            coeffs = []
            all_ones = True
            for index, coeff in zip(proto.vars, proto.coeffs):
                variables.append(IntVar(model, index, None))
                coeffs.append(coeff)
                if not cmh.is_one(coeff):
                    all_ones = False
            if all_ones:
                return _SumArray(variables, offset)
            else:
                return _WeightedSum(variables, coeffs, offset)

    def get_integer_var_value_map(self) -> Tuple[Dict['IntVar', IntegralT], int]:
        """Scans the expression, and returns (var_coef_map, constant)."""
        coeffs = collections.defaultdict(int)
        constant = 0
        to_process: List[Tuple[LinearExprT, IntegralT]] = [(self, 1)]
        while to_process:
            expr, coeff = to_process.pop()
            if isinstance(expr, numbers.Integral):
                constant += coeff * int(expr)
            elif isinstance(expr, _ProductCst):
                to_process.append((expr.expression(), coeff * expr.coefficient()))
            elif isinstance(expr, _Sum):
                to_process.append((expr.left(), coeff))
                to_process.append((expr.right(), coeff))
            elif isinstance(expr, _SumArray):
                for e in expr.expressions():
                    to_process.append((e, coeff))
                constant += expr.constant() * coeff
            elif isinstance(expr, _WeightedSum):
                for e, c in zip(expr.expressions(), expr.coefficients()):
                    to_process.append((e, coeff * c))
                constant += expr.constant() * coeff
            elif isinstance(expr, IntVar):
                coeffs[expr] += coeff
            elif isinstance(expr, _NotBooleanVariable):
                constant += coeff
                coeffs[expr.negated()] -= coeff
            else:
                raise TypeError('Unrecognized linear expression: ' + str(expr))
        return (coeffs, constant)

    def get_float_var_value_map(self) -> Tuple[Dict['IntVar', float], float, bool]:
        """Scans the expression. Returns (var_coef_map, constant, is_integer)."""
        coeffs = {}
        constant = 0
        to_process: List[Tuple[LinearExprT, Union[IntegralT, float]]] = [(self, 1)]
        while to_process:
            expr, coeff = to_process.pop()
            if isinstance(expr, numbers.Integral):
                constant += coeff * int(expr)
            elif isinstance(expr, numbers.Number):
                constant += coeff * float(expr)
            elif isinstance(expr, _ProductCst):
                to_process.append((expr.expression(), coeff * expr.coefficient()))
            elif isinstance(expr, _Sum):
                to_process.append((expr.left(), coeff))
                to_process.append((expr.right(), coeff))
            elif isinstance(expr, _SumArray):
                for e in expr.expressions():
                    to_process.append((e, coeff))
                constant += expr.constant() * coeff
            elif isinstance(expr, _WeightedSum):
                for e, c in zip(expr.expressions(), expr.coefficients()):
                    to_process.append((e, coeff * c))
                constant += expr.constant() * coeff
            elif isinstance(expr, IntVar):
                if expr in coeffs:
                    coeffs[expr] += coeff
                else:
                    coeffs[expr] = coeff
            elif isinstance(expr, _NotBooleanVariable):
                constant += coeff
                if expr.negated() in coeffs:
                    coeffs[expr.negated()] -= coeff
                else:
                    coeffs[expr.negated()] = -coeff
            else:
                raise TypeError('Unrecognized linear expression: ' + str(expr))
        is_integer = isinstance(constant, numbers.Integral)
        if is_integer:
            for coeff in coeffs.values():
                if not isinstance(coeff, numbers.Integral):
                    is_integer = False
                    break
        return (coeffs, constant, is_integer)

    def __hash__(self) -> int:
        return object.__hash__(self)

    def __abs__(self) -> NoReturn:
        raise NotImplementedError('calling abs() on a linear expression is not supported, please use CpModel.add_abs_equality')

    @overload
    def __add__(self, arg: LinearExprT) -> LinearExprT:
        ...

    @overload
    def __add__(self, arg: ObjLinearExprT) -> ObjLinearExprT:
        ...

    def __add__(self, arg):
        if cmh.is_zero(arg):
            return self
        return _Sum(self, arg)

    @overload
    def __radd__(self, arg: LinearExprT) -> LinearExprT:
        ...

    @overload
    def __radd__(self, arg: ObjLinearExprT) -> ObjLinearExprT:
        ...

    def __radd__(self, arg):
        if cmh.is_zero(arg):
            return self
        return _Sum(self, arg)

    @overload
    def __sub__(self, arg: LinearExprT) -> LinearExprT:
        ...

    @overload
    def __sub__(self, arg: ObjLinearExprT) -> ObjLinearExprT:
        ...

    def __sub__(self, arg):
        if cmh.is_zero(arg):
            return self
        if isinstance(arg, numbers.Number):
            arg = cmh.assert_is_a_number(arg)
            return _Sum(self, -arg)
        else:
            return _Sum(self, -arg)

    @overload
    def __rsub__(self, arg: LinearExprT) -> LinearExprT:
        ...

    @overload
    def __rsub__(self, arg: ObjLinearExprT) -> ObjLinearExprT:
        ...

    def __rsub__(self, arg):
        return _Sum(-self, arg)

    @overload
    def __mul__(self, arg: LinearExprT) -> LinearExprT:
        ...

    @overload
    def __mul__(self, arg: ObjLinearExprT) -> ObjLinearExprT:
        ...

    def __mul__(self, arg):
        arg = cmh.assert_is_a_number(arg)
        if cmh.is_one(arg):
            return self
        elif cmh.is_zero(arg):
            return 0
        return _ProductCst(self, arg)

    @overload
    def __rmul__(self, arg: LinearExprT) -> LinearExprT:
        ...

    @overload
    def __rmul__(self, arg: ObjLinearExprT) -> ObjLinearExprT:
        ...

    def __rmul__(self, arg):
        arg = cmh.assert_is_a_number(arg)
        if cmh.is_one(arg):
            return self
        elif cmh.is_zero(arg):
            return 0
        return _ProductCst(self, arg)

    def __div__(self, _) -> NoReturn:
        raise NotImplementedError('calling / on a linear expression is not supported, please use CpModel.add_division_equality')

    def __truediv__(self, _) -> NoReturn:
        raise NotImplementedError('calling // on a linear expression is not supported, please use CpModel.add_division_equality')

    def __mod__(self, _) -> NoReturn:
        raise NotImplementedError('calling %% on a linear expression is not supported, please use CpModel.add_modulo_equality')

    def __pow__(self, _) -> NoReturn:
        raise NotImplementedError('calling ** on a linear expression is not supported, please use CpModel.add_multiplication_equality')

    def __lshift__(self, _) -> NoReturn:
        raise NotImplementedError('calling left shift on a linear expression is not supported')

    def __rshift__(self, _) -> NoReturn:
        raise NotImplementedError('calling right shift on a linear expression is not supported')

    def __and__(self, _) -> NoReturn:
        raise NotImplementedError('calling and on a linear expression is not supported, please use CpModel.add_bool_and')

    def __or__(self, _) -> NoReturn:
        raise NotImplementedError('calling or on a linear expression is not supported, please use CpModel.add_bool_or')

    def __xor__(self, _) -> NoReturn:
        raise NotImplementedError('calling xor on a linear expression is not supported, please use CpModel.add_bool_xor')

    def __neg__(self) -> LinearExprT:
        return _ProductCst(self, -1)

    def __bool__(self) -> NoReturn:
        raise NotImplementedError('Evaluating a LinearExpr instance as a Boolean is not implemented.')

    def __eq__(self, arg: LinearExprT) -> BoundedLinearExprT:
        if arg is None:
            return False
        if isinstance(arg, numbers.Integral):
            arg = cmh.assert_is_int64(arg)
            return BoundedLinearExpression(self, [arg, arg])
        else:
            return BoundedLinearExpression(self - arg, [0, 0])

    def __ge__(self, arg: LinearExprT) -> BoundedLinearExprT:
        if isinstance(arg, numbers.Integral):
            arg = cmh.assert_is_int64(arg)
            return BoundedLinearExpression(self, [arg, INT_MAX])
        else:
            return BoundedLinearExpression(self - arg, [0, INT_MAX])

    def __le__(self, arg: LinearExprT) -> BoundedLinearExprT:
        if isinstance(arg, numbers.Integral):
            arg = cmh.assert_is_int64(arg)
            return BoundedLinearExpression(self, [INT_MIN, arg])
        else:
            return BoundedLinearExpression(self - arg, [INT_MIN, 0])

    def __lt__(self, arg: LinearExprT) -> BoundedLinearExprT:
        if isinstance(arg, numbers.Integral):
            arg = cmh.assert_is_int64(arg)
            if arg == INT_MIN:
                raise ArithmeticError('< INT_MIN is not supported')
            return BoundedLinearExpression(self, [INT_MIN, arg - 1])
        else:
            return BoundedLinearExpression(self - arg, [INT_MIN, -1])

    def __gt__(self, arg: LinearExprT) -> BoundedLinearExprT:
        if isinstance(arg, numbers.Integral):
            arg = cmh.assert_is_int64(arg)
            if arg == INT_MAX:
                raise ArithmeticError('> INT_MAX is not supported')
            return BoundedLinearExpression(self, [arg + 1, INT_MAX])
        else:
            return BoundedLinearExpression(self - arg, [1, INT_MAX])

    def __ne__(self, arg: LinearExprT) -> BoundedLinearExprT:
        if arg is None:
            return True
        if isinstance(arg, numbers.Integral):
            arg = cmh.assert_is_int64(arg)
            if arg == INT_MAX:
                return BoundedLinearExpression(self, [INT_MIN, INT_MAX - 1])
            elif arg == INT_MIN:
                return BoundedLinearExpression(self, [INT_MIN + 1, INT_MAX])
            else:
                return BoundedLinearExpression(self, [INT_MIN, arg - 1, arg + 1, INT_MAX])
        else:
            return BoundedLinearExpression(self - arg, [INT_MIN, -1, 1, INT_MAX])

    @classmethod
    def Sum(cls, expressions: Sequence[LinearExprT]) -> LinearExprT:
        """Creates the expression sum(expressions)."""
        return cls.sum(expressions)

    @overload
    @classmethod
    def WeightedSum(cls, expressions: Sequence[LinearExprT], coefficients: Sequence[IntegralT]) -> LinearExprT:
        ...

    @overload
    @classmethod
    def WeightedSum(cls, expressions: Sequence[ObjLinearExprT], coefficients: Sequence[NumberT]) -> ObjLinearExprT:
        ...

    @classmethod
    def WeightedSum(cls, expressions, coefficients):
        """Creates the expression sum(expressions[i] * coefficients[i])."""
        return cls.weighted_sum(expressions, coefficients)

    @overload
    @classmethod
    def Term(cls, expressions: LinearExprT, coefficients: IntegralT) -> LinearExprT:
        ...

    @overload
    @classmethod
    def Term(cls, expressions: ObjLinearExprT, coefficients: NumberT) -> ObjLinearExprT:
        ...

    @classmethod
    def Term(cls, expression, coefficient):
        """Creates `expression * coefficient`."""
        return cls.term(expression, coefficient)