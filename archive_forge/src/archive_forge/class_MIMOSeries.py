from typing import Type
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import ImmutableMatrix, eye
from sympy.matrices.expressions import MatMul, MatAdd
from sympy.polys import Poly, rootof
from sympy.polys.polyroots import roots
from sympy.polys.polytools import (cancel, degree)
from sympy.series import limit
from mpmath.libmp.libmpf import prec_to_dps
class MIMOSeries(MIMOLinearTimeInvariant):
    """
    A class for representing a series configuration of MIMO systems.

    Parameters
    ==========

    args : MIMOLinearTimeInvariant
        MIMO systems in a series configuration.
    evaluate : Boolean, Keyword
        When passed ``True``, returns the equivalent
        ``MIMOSeries(*args).doit()``. Set to ``False`` by default.

    Raises
    ======

    ValueError
        When no argument is passed.

        ``var`` attribute is not same for every system.

        ``num_outputs`` of the MIMO system is not equal to the
        ``num_inputs`` of its adjacent MIMO system. (Matrix
        multiplication constraint, basically)
    TypeError
        Any of the passed ``*args`` has unsupported type

        A combination of SISO and MIMO systems is
        passed. There should be homogeneity in the
        type of systems passed, MIMO in this case.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import MIMOSeries, TransferFunctionMatrix
    >>> from sympy import Matrix, pprint
    >>> mat_a = Matrix([[5*s], [5]])  # 2 Outputs 1 Input
    >>> mat_b = Matrix([[5, 1/(6*s**2)]])  # 1 Output 2 Inputs
    >>> mat_c = Matrix([[1, s], [5/s, 1]])  # 2 Outputs 2 Inputs
    >>> tfm_a = TransferFunctionMatrix.from_Matrix(mat_a, s)
    >>> tfm_b = TransferFunctionMatrix.from_Matrix(mat_b, s)
    >>> tfm_c = TransferFunctionMatrix.from_Matrix(mat_c, s)
    >>> MIMOSeries(tfm_c, tfm_b, tfm_a)
    MIMOSeries(TransferFunctionMatrix(((TransferFunction(1, 1, s), TransferFunction(s, 1, s)), (TransferFunction(5, s, s), TransferFunction(1, 1, s)))), TransferFunctionMatrix(((TransferFunction(5, 1, s), TransferFunction(1, 6*s**2, s)),)), TransferFunctionMatrix(((TransferFunction(5*s, 1, s),), (TransferFunction(5, 1, s),))))
    >>> pprint(_, use_unicode=False)  #  For Better Visualization
    [5*s]                 [1  s]
    [---]    [5   1  ]    [-  -]
    [ 1 ]    [-  ----]    [1  1]
    [   ]   *[1     2]   *[    ]
    [ 5 ]    [   6*s ]{t} [5  1]
    [ - ]                 [-  -]
    [ 1 ]{t}              [s  1]{t}
    >>> MIMOSeries(tfm_c, tfm_b, tfm_a).doit()
    TransferFunctionMatrix(((TransferFunction(150*s**4 + 25*s, 6*s**3, s), TransferFunction(150*s**4 + 5*s, 6*s**2, s)), (TransferFunction(150*s**3 + 25, 6*s**3, s), TransferFunction(150*s**3 + 5, 6*s**2, s))))
    >>> pprint(_, use_unicode=False)  # (2 Inputs -A-> 2 Outputs) -> (2 Inputs -B-> 1 Output) -> (1 Input -C-> 2 Outputs) is equivalent to (2 Inputs -Series Equivalent-> 2 Outputs).
    [     4              4      ]
    [150*s  + 25*s  150*s  + 5*s]
    [-------------  ------------]
    [        3             2    ]
    [     6*s           6*s     ]
    [                           ]
    [      3              3     ]
    [ 150*s  + 25    150*s  + 5 ]
    [ -----------    ---------- ]
    [        3             2    ]
    [     6*s           6*s     ]{t}

    Notes
    =====

    All the transfer function matrices should use the same complex variable ``var`` of the Laplace transform.

    ``MIMOSeries(A, B)`` is not equivalent to ``A*B``. It is always in the reverse order, that is ``B*A``.

    See Also
    ========

    Series, MIMOParallel

    """

    def __new__(cls, *args, evaluate=False):
        cls._check_args(args)
        if _mat_mul_compatible(*args):
            obj = super().__new__(cls, *args)
        else:
            raise ValueError('Number of input signals do not match the number of output signals of adjacent systems for some args.')
        return obj.doit() if evaluate else obj

    @property
    def var(self):
        """
        Returns the complex variable used by all the transfer functions.

        Examples
        ========

        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, MIMOSeries, TransferFunctionMatrix
        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)
        >>> G2 = TransferFunction(p, 4 - p, p)
        >>> G3 = TransferFunction(0, p**4 - 1, p)
        >>> tfm_1 = TransferFunctionMatrix([[G1, G2, G3]])
        >>> tfm_2 = TransferFunctionMatrix([[G1], [G2], [G3]])
        >>> MIMOSeries(tfm_2, tfm_1).var
        p

        """
        return self.args[0].var

    @property
    def num_inputs(self):
        """Returns the number of input signals of the series system."""
        return self.args[0].num_inputs

    @property
    def num_outputs(self):
        """Returns the number of output signals of the series system."""
        return self.args[-1].num_outputs

    @property
    def shape(self):
        """Returns the shape of the equivalent MIMO system."""
        return (self.num_outputs, self.num_inputs)

    def doit(self, cancel=False, **kwargs):
        """
        Returns the resultant transfer function matrix obtained after evaluating
        the MIMO systems arranged in a series configuration.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, MIMOSeries, TransferFunctionMatrix
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
        >>> tfm1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf2]])
        >>> tfm2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf1]])
        >>> MIMOSeries(tfm2, tfm1).doit()
        TransferFunctionMatrix(((TransferFunction(2*(-p + s)*(s**3 - 2)*(a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)**2*(s**4 + 5*s + 6)**2, s), TransferFunction((-p + s)**2*(s**3 - 2)*(a*p**2 + b*s) + (-p + s)*(a*p**2 + b*s)**2*(s**4 + 5*s + 6), (-p + s)**3*(s**4 + 5*s + 6), s)), (TransferFunction((-p + s)*(s**3 - 2)**2*(s**4 + 5*s + 6) + (s**3 - 2)*(a*p**2 + b*s)*(s**4 + 5*s + 6)**2, (-p + s)*(s**4 + 5*s + 6)**3, s), TransferFunction(2*(s**3 - 2)*(a*p**2 + b*s), (-p + s)*(s**4 + 5*s + 6), s))))

        """
        _arg = (arg.doit()._expr_mat for arg in reversed(self.args))
        if cancel:
            res = MatMul(*_arg, evaluate=True)
            return TransferFunctionMatrix.from_Matrix(res, self.var)
        _dummy_args, _dummy_dict = _dummify_args(_arg, self.var)
        res = MatMul(*_dummy_args, evaluate=True)
        temp_tfm = TransferFunctionMatrix.from_Matrix(res, self.var)
        return temp_tfm.subs(_dummy_dict)

    def _eval_rewrite_as_TransferFunctionMatrix(self, *args, **kwargs):
        return self.doit()

    @_check_other_MIMO
    def __add__(self, other):
        if isinstance(other, MIMOParallel):
            arg_list = list(other.args)
            return MIMOParallel(self, *arg_list)
        return MIMOParallel(self, other)
    __radd__ = __add__

    @_check_other_MIMO
    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    @_check_other_MIMO
    def __mul__(self, other):
        if isinstance(other, MIMOSeries):
            self_arg_list = list(self.args)
            other_arg_list = list(other.args)
            return MIMOSeries(*other_arg_list, *self_arg_list)
        arg_list = list(self.args)
        return MIMOSeries(other, *arg_list)

    def __neg__(self):
        arg_list = list(self.args)
        arg_list[0] = -arg_list[0]
        return MIMOSeries(*arg_list)