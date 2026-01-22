import abc
from typing import Generator, Type, Union
import numpy as np
import pandas
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.dtypes.common import (
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import ColNameCodec, to_arrow_type
class BaseExpr(abc.ABC):
    """
    An abstract base class for expression tree node.

    An expression tree is used to describe how a single column of a dataframe
    is computed.

    Each node can belong to multiple trees and therefore should be immutable
    until proven to have no parent nodes (e.g. by making a copy).

    Attributes
    ----------
    operands : list of BaseExpr, optional
        Holds child nodes. Leaf nodes shouldn't have `operands` attribute.
    """
    binary_operations = {'add': '+', 'sub': '-', 'mul': '*', 'mod': 'MOD', 'floordiv': '//', 'truediv': '/', 'pow': 'POWER', 'eq': '=', 'ge': '>=', 'gt': '>', 'le': '<=', 'lt': '<', 'ne': '<>', 'and': 'AND', 'or': 'OR'}
    preserve_dtype_math_ops = {'add', 'sub', 'mul', 'mod', 'floordiv', 'pow'}
    promote_to_float_math_ops = {'truediv'}

    def eq(self, other):
        """
        Build an equality comparison of `self` with `other`.

        Parameters
        ----------
        other : BaseExpr or scalar
            An operand to compare with.

        Returns
        -------
        BaseExpr
            The resulting comparison expression.
        """
        return self.cmp('=', other)

    def le(self, other):
        """
        Build a less or equal comparison with `other`.

        Parameters
        ----------
        other : BaseExpr or scalar
            An operand to compare with.

        Returns
        -------
        BaseExpr
            The resulting comparison expression.
        """
        return self.cmp('<=', other)

    def ge(self, other):
        """
        Build a greater or equal comparison with `other`.

        Parameters
        ----------
        other : BaseExpr or scalar
            An operand to compare with.

        Returns
        -------
        BaseExpr
            The resulting comparison expression.
        """
        return self.cmp('>=', other)

    def cmp(self, op, other):
        """
        Build a comparison expression with `other`.

        Parameters
        ----------
        op : str
            A comparison operation.
        other : BaseExpr or scalar
            An operand to compare with.

        Returns
        -------
        BaseExpr
            The resulting comparison expression.
        """
        if not isinstance(other, BaseExpr):
            other = LiteralExpr(other)
        return OpExpr(op, [self, other], _get_dtype(bool))

    def cast(self, res_type):
        """
        Build a cast expression.

        Parameters
        ----------
        res_type : dtype
            A data type to cast to.

        Returns
        -------
        BaseExpr
            The cast expression.
        """
        if is_float_dtype(self._dtype) and is_integer_dtype(res_type):
            return self.floor()
        new_expr = OpExpr('CAST', [self], res_type)
        return new_expr

    def is_null(self):
        """
        Build a NULL check expression.

        Returns
        -------
        BaseExpr
            The NULL check expression.
        """
        new_expr = OpExpr('IS NULL', [self], _get_dtype(bool))
        return new_expr

    def is_not_null(self):
        """
        Build a NOT NULL check expression.

        Returns
        -------
        BaseExpr
            The NOT NULL check expression.
        """
        new_expr = OpExpr('IS NOT NULL', [self], _get_dtype(bool))
        return new_expr

    def bin_op(self, other, op_name):
        """
        Build a binary operation expression.

        Parameters
        ----------
        other : BaseExpr
            The second operand.
        op_name : str
            A binary operation name.

        Returns
        -------
        BaseExpr
            The resulting binary operation expression.
        """
        if op_name not in self.binary_operations:
            raise NotImplementedError(f'unsupported binary operation {op_name}')
        if is_cmp_op(op_name):
            return self._cmp_op(other, op_name)
        if op_name == 'truediv':
            if is_integer_dtype(self._dtype) and is_integer_dtype(other._dtype):
                other = other.cast(_get_dtype(float))
        res_type = self._get_bin_op_res_type(op_name, self._dtype, other._dtype)
        new_expr = OpExpr(self.binary_operations[op_name], [self, other], res_type)
        if op_name == 'floordiv' and (not is_integer_dtype(res_type)):
            return new_expr.floor()
        return new_expr

    def add(self, other):
        """
        Build an add expression.

        Parameters
        ----------
        other : BaseExpr
            The second operand.

        Returns
        -------
        BaseExpr
            The resulting add expression.
        """
        return self.bin_op(other, 'add')

    def sub(self, other):
        """
        Build a sub expression.

        Parameters
        ----------
        other : BaseExpr
            The second operand.

        Returns
        -------
        BaseExpr
            The resulting sub expression.
        """
        return self.bin_op(other, 'sub')

    def mul(self, other):
        """
        Build a mul expression.

        Parameters
        ----------
        other : BaseExpr
            The second operand.

        Returns
        -------
        BaseExpr
            The resulting mul expression.
        """
        return self.bin_op(other, 'mul')

    def mod(self, other):
        """
        Build a mod expression.

        Parameters
        ----------
        other : BaseExpr
            The second operand.

        Returns
        -------
        BaseExpr
            The resulting mod expression.
        """
        return self.bin_op(other, 'mod')

    def truediv(self, other):
        """
        Build a truediv expression.

        The result always has float data type.

        Parameters
        ----------
        other : BaseExpr
            The second operand.

        Returns
        -------
        BaseExpr
            The resulting truediv expression.
        """
        return self.bin_op(other, 'truediv')

    def floordiv(self, other):
        """
        Build a floordiv expression.

        The result always has an integer data type.

        Parameters
        ----------
        other : BaseExpr
            The second operand.

        Returns
        -------
        BaseExpr
            The resulting floordiv expression.
        """
        return self.bin_op(other, 'floordiv')

    def pow(self, other):
        """
        Build a power expression.

        Parameters
        ----------
        other : BaseExpr
            The power operand.

        Returns
        -------
        BaseExpr
            The resulting power expression.
        """
        return self.bin_op(other, 'pow')

    def floor(self):
        """
        Build a floor expression.

        Returns
        -------
        BaseExpr
            The resulting floor expression.
        """
        return OpExpr('FLOOR', [self], _get_dtype(int))

    def invert(self) -> 'OpExpr':
        """
        Build a bitwise inverse expression.

        Returns
        -------
        OpExpr
            The resulting bitwise inverse expression.
        """
        return OpExpr('BIT_NOT', [self], self._dtype)

    def _cmp_op(self, other, op_name):
        """
        Build a comparison expression.

        Parameters
        ----------
        other : BaseExpr
            A value to compare with.
        op_name : str
            The comparison operation name.

        Returns
        -------
        BaseExpr
            The resulting comparison expression.
        """
        lhs_dtype_class = self._get_dtype_cmp_class(self._dtype)
        rhs_dtype_class = self._get_dtype_cmp_class(other._dtype)
        res_dtype = _get_dtype(bool)
        if lhs_dtype_class != rhs_dtype_class:
            if op_name == 'eq' or op_name == 'ne':
                return LiteralExpr(op_name == 'ne')
            else:
                raise TypeError(f'Invalid comparison between {self._dtype} and {other._dtype}')
        else:
            cmp = OpExpr(self.binary_operations[op_name], [self, other], res_dtype)
            return build_if_then_else(self.is_null(), LiteralExpr(op_name == 'ne'), cmp, res_dtype)

    @staticmethod
    def _get_dtype_cmp_class(dtype):
        """
        Get a comparison class name for specified data type.

        Values of different comparison classes cannot be compared.

        Parameters
        ----------
        dtype : dtype
            A data type of a compared value.

        Returns
        -------
        str
            The comparison class name.
        """
        if is_numeric_dtype(dtype) or is_bool_dtype(dtype):
            return 'numeric'
        if is_string_dtype(dtype) or isinstance(dtype, pandas.CategoricalDtype):
            return 'string'
        if is_datetime64_any_dtype(dtype):
            return 'datetime'
        return 'other'

    def _get_bin_op_res_type(self, op_name, lhs_dtype, rhs_dtype):
        """
        Return the result data type for a binary operation.

        Parameters
        ----------
        op_name : str
            A binary operation name.
        lhs_dtype : dtype
            A left operand's type.
        rhs_dtype : dtype
            A right operand's type.

        Returns
        -------
        dtype
        """
        if op_name in self.preserve_dtype_math_ops:
            return _get_common_dtype(lhs_dtype, rhs_dtype)
        elif op_name in self.promote_to_float_math_ops:
            return _get_dtype(float)
        elif is_cmp_op(op_name):
            return _get_dtype(bool)
        elif is_logical_op(op_name):
            return _get_dtype(bool)
        else:
            raise NotImplementedError(f'unsupported binary operation {op_name}')

    @abc.abstractmethod
    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        BaseExpr
        """
        pass

    def nested_expressions(self) -> Generator[Type['BaseExpr'], Type['BaseExpr'], Type['BaseExpr']]:
        """
        Return a generator that allows to iterate over and replace the nested expressions.

        If the generator receives a new expression, it creates a copy of `self` and
        replaces the expression in the copy. The copy is returned to the sender.

        Returns
        -------
        Generator
        """
        expr = self
        if (operands := getattr(self, 'operands', None)):
            for i, op in enumerate(operands):
                new_op = (yield op)
                if new_op is not None:
                    if new_op is not op:
                        if expr is self:
                            expr = self.copy()
                        expr.operands[i] = new_op
                    yield expr
        return expr

    def collect_frames(self, frames):
        """
        Recursively collect all frames participating in the expression.

        Collected frames are put into the `frames` set. Default implementation
        collects frames from the operands of the expression. Derived classes
        directly holding frames should provide their own implementations.

        Parameters
        ----------
        frames : set
            Output set of collected frames.
        """
        for expr in self.nested_expressions():
            expr.collect_frames(frames)

    def translate_input(self, mapper):
        """
        Make a deep copy of the expression translating input nodes using `mapper`.

        The default implementation builds a copy and recursively run
        translation for all its operands. For leaf expressions
        `_translate_input` is called.

        Parameters
        ----------
        mapper : InputMapper
            A mapper to use for input columns translation.

        Returns
        -------
        BaseExpr
            The expression copy with translated input columns.
        """
        res = None
        gen = self.nested_expressions()
        for expr in gen:
            res = gen.send(expr.translate_input(mapper))
        return self._translate_input(mapper) if res is None else res

    def _translate_input(self, mapper):
        """
        Make a deep copy of the expression translating input nodes using `mapper`.

        Called by default translator for leaf nodes. Method should be overriden
        by derived classes holding input references.

        Parameters
        ----------
        mapper : InputMapper
            A mapper to use for input columns translation.

        Returns
        -------
        BaseExpr
            The expression copy with translated input columns.
        """
        return self

    def fold(self):
        """
        Fold the operands.

        This operation is used by `TransformNode` when translating to base.

        Returns
        -------
        BaseExpr
        """
        res = self
        gen = self.nested_expressions()
        for expr in gen:
            res = gen.send(expr.fold())
        return res

    def can_execute_hdk(self) -> bool:
        """
        Check for possibility of HDK execution.

        Check if the computation can be executed using an HDK query.

        Returns
        -------
        bool
        """
        return True

    def can_execute_arrow(self) -> bool:
        """
        Check for possibility of Arrow execution.

        Check if the computation can be executed using
        the Arrow API instead of HDK query.

        Returns
        -------
        bool
        """
        return False

    def execute_arrow(self, table: pa.Table) -> pa.ChunkedArray:
        """
        Compute the column data using the Arrow API.

        Parameters
        ----------
        table : pa.Table

        Returns
        -------
        pa.ChunkedArray
        """
        raise RuntimeError(f'Arrow execution is not supported by {type(self)}')