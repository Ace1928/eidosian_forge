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
class MIMOFeedback(MIMOLinearTimeInvariant):
    """
    A class for representing closed-loop feedback interconnection between two
    MIMO input/output systems.

    Parameters
    ==========

    sys1 : MIMOSeries, TransferFunctionMatrix
        The MIMO system placed on the feedforward path.
    sys2 : MIMOSeries, TransferFunctionMatrix
        The system placed on the feedback path
        (often a feedback controller).
    sign : int, optional
        The sign of feedback. Can either be ``1``
        (for positive feedback) or ``-1`` (for negative feedback).
        Default value is `-1`.

    Raises
    ======

    ValueError
        When ``sys1`` and ``sys2`` are not using the
        same complex variable of the Laplace transform.

        Forward path model should have an equal number of inputs/outputs
        to the feedback path outputs/inputs.

        When product of ``sys1`` and ``sys2`` is not a square matrix.

        When the equivalent MIMO system is not invertible.

    TypeError
        When either ``sys1`` or ``sys2`` is not a ``MIMOSeries`` or a
        ``TransferFunctionMatrix`` object.

    Examples
    ========

    >>> from sympy import Matrix, pprint
    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunctionMatrix, MIMOFeedback
    >>> plant_mat = Matrix([[1, 1/s], [0, 1]])
    >>> controller_mat = Matrix([[10, 0], [0, 10]])  # Constant Gain
    >>> plant = TransferFunctionMatrix.from_Matrix(plant_mat, s)
    >>> controller = TransferFunctionMatrix.from_Matrix(controller_mat, s)
    >>> feedback = MIMOFeedback(plant, controller)  # Negative Feedback (default)
    >>> pprint(feedback, use_unicode=False)
    /    [1  1]    [10  0 ]   \\-1   [1  1]
    |    [-  -]    [--  - ]   |     [-  -]
    |    [1  s]    [1   1 ]   |     [1  s]
    |I + [    ]   *[      ]   |   * [    ]
    |    [0  1]    [0   10]   |     [0  1]
    |    [-  -]    [-   --]   |     [-  -]
    \\    [1  1]{t} [1   1 ]{t}/     [1  1]{t}

    To get the equivalent system matrix, use either ``doit`` or ``rewrite`` method.

    >>> pprint(feedback.doit(), use_unicode=False)
    [1     1  ]
    [--  -----]
    [11  121*s]
    [         ]
    [0    1   ]
    [-    --  ]
    [1    11  ]{t}

    To negate the ``MIMOFeedback`` object, use ``-`` operator.

    >>> neg_feedback = -feedback
    >>> pprint(neg_feedback.doit(), use_unicode=False)
    [-1    -1  ]
    [---  -----]
    [ 11  121*s]
    [          ]
    [ 0    -1  ]
    [ -    --- ]
    [ 1     11 ]{t}

    See Also
    ========

    Feedback, MIMOSeries, MIMOParallel

    """

    def __new__(cls, sys1, sys2, sign=-1):
        if not (isinstance(sys1, (TransferFunctionMatrix, MIMOSeries)) and isinstance(sys2, (TransferFunctionMatrix, MIMOSeries))):
            raise TypeError('Unsupported type for `sys1` or `sys2` of MIMO Feedback.')
        if sys1.num_inputs != sys2.num_outputs or sys1.num_outputs != sys2.num_inputs:
            raise ValueError('Product of `sys1` and `sys2` must yield a square matrix.')
        if sign not in (-1, 1):
            raise ValueError('Unsupported type for feedback. `sign` arg should either be 1 (positive feedback loop) or -1 (negative feedback loop).')
        if not _is_invertible(sys1, sys2, sign):
            raise ValueError('Non-Invertible system inputted.')
        if sys1.var != sys2.var:
            raise ValueError('Both `sys1` and `sys2` should be using the same complex variable.')
        return super().__new__(cls, sys1, sys2, _sympify(sign))

    @property
    def sys1(self):
        """
        Returns the system placed on the feedforward path of the MIMO feedback interconnection.

        Examples
        ========

        >>> from sympy import pprint
        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback
        >>> tf1 = TransferFunction(s**2 + s + 1, s**2 - s + 1, s)
        >>> tf2 = TransferFunction(1, s, s)
        >>> tf3 = TransferFunction(1, 1, s)
        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
        >>> sys2 = TransferFunctionMatrix([[tf3, tf3], [tf3, tf2]])
        >>> F_1 = MIMOFeedback(sys1, sys2, 1)
        >>> F_1.sys1
        TransferFunctionMatrix(((TransferFunction(s**2 + s + 1, s**2 - s + 1, s), TransferFunction(1, s, s)), (TransferFunction(1, s, s), TransferFunction(s**2 + s + 1, s**2 - s + 1, s))))
        >>> pprint(_, use_unicode=False)
        [ 2                    ]
        [s  + s + 1      1     ]
        [----------      -     ]
        [ 2              s     ]
        [s  - s + 1            ]
        [                      ]
        [             2        ]
        [    1       s  + s + 1]
        [    -       ----------]
        [    s        2        ]
        [            s  - s + 1]{t}

        """
        return self.args[0]

    @property
    def sys2(self):
        """
        Returns the feedback controller of the MIMO feedback interconnection.

        Examples
        ========

        >>> from sympy import pprint
        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback
        >>> tf1 = TransferFunction(s**2, s**3 - s + 1, s)
        >>> tf2 = TransferFunction(1, s, s)
        >>> tf3 = TransferFunction(1, 1, s)
        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
        >>> sys2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf2]])
        >>> F_1 = MIMOFeedback(sys1, sys2)
        >>> F_1.sys2
        TransferFunctionMatrix(((TransferFunction(s**2, s**3 - s + 1, s), TransferFunction(1, 1, s)), (TransferFunction(1, 1, s), TransferFunction(1, s, s))))
        >>> pprint(_, use_unicode=False)
        [     2       ]
        [    s       1]
        [----------  -]
        [ 3          1]
        [s  - s + 1   ]
        [             ]
        [    1       1]
        [    -       -]
        [    1       s]{t}

        """
        return self.args[1]

    @property
    def var(self):
        """
        Returns the complex variable of the Laplace transform used by all
        the transfer functions involved in the MIMO feedback loop.

        Examples
        ========

        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback
        >>> tf1 = TransferFunction(p, 1 - p, p)
        >>> tf2 = TransferFunction(1, p, p)
        >>> tf3 = TransferFunction(1, 1, p)
        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
        >>> sys2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf2]])
        >>> F_1 = MIMOFeedback(sys1, sys2, 1)  # Positive feedback
        >>> F_1.var
        p

        """
        return self.sys1.var

    @property
    def sign(self):
        """
        Returns the type of feedback interconnection of two models. ``1``
        for Positive and ``-1`` for Negative.
        """
        return self.args[2]

    @property
    def sensitivity(self):
        """
        Returns the sensitivity function matrix of the feedback loop.

        Sensitivity of a closed-loop system is the ratio of change
        in the open loop gain to the change in the closed loop gain.

        .. note::
            This method would not return the complementary
            sensitivity function.

        Examples
        ========

        >>> from sympy import pprint
        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback
        >>> tf1 = TransferFunction(p, 1 - p, p)
        >>> tf2 = TransferFunction(1, p, p)
        >>> tf3 = TransferFunction(1, 1, p)
        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
        >>> sys2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf2]])
        >>> F_1 = MIMOFeedback(sys1, sys2, 1)  # Positive feedback
        >>> F_2 = MIMOFeedback(sys1, sys2)  # Negative feedback
        >>> pprint(F_1.sensitivity, use_unicode=False)
        [   4      3      2               5      4      2           ]
        [- p  + 3*p  - 4*p  + 3*p - 1    p  - 2*p  + 3*p  - 3*p + 1 ]
        [----------------------------  -----------------------------]
        [  4      3      2              5      4      3      2      ]
        [ p  + 3*p  - 8*p  + 8*p - 3   p  + 3*p  - 8*p  + 8*p  - 3*p]
        [                                                           ]
        [       4    3    2                  3      2               ]
        [      p  - p  - p  + p           3*p  - 6*p  + 4*p - 1     ]
        [ --------------------------    --------------------------  ]
        [  4      3      2               4      3      2            ]
        [ p  + 3*p  - 8*p  + 8*p - 3    p  + 3*p  - 8*p  + 8*p - 3  ]
        >>> pprint(F_2.sensitivity, use_unicode=False)
        [ 4      3      2           5      4      2          ]
        [p  - 3*p  + 2*p  + p - 1  p  - 2*p  + 3*p  - 3*p + 1]
        [------------------------  --------------------------]
        [   4      3                   5      4      2       ]
        [  p  - 3*p  + 2*p - 1        p  - 3*p  + 2*p  - p   ]
        [                                                    ]
        [     4    3    2               4      3             ]
        [    p  - p  - p  + p        2*p  - 3*p  + 2*p - 1   ]
        [  -------------------       ---------------------   ]
        [   4      3                   4      3              ]
        [  p  - 3*p  + 2*p - 1        p  - 3*p  + 2*p - 1    ]

        """
        _sys1_mat = self.sys1.doit()._expr_mat
        _sys2_mat = self.sys2.doit()._expr_mat
        return (eye(self.sys1.num_inputs) - self.sign * _sys1_mat * _sys2_mat).inv()

    def doit(self, cancel=True, expand=False, **hints):
        """
        Returns the resultant transfer function matrix obtained by the
        feedback interconnection.

        Examples
        ========

        >>> from sympy import pprint
        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback
        >>> tf1 = TransferFunction(s, 1 - s, s)
        >>> tf2 = TransferFunction(1, s, s)
        >>> tf3 = TransferFunction(5, 1, s)
        >>> tf4 = TransferFunction(s - 1, s, s)
        >>> tf5 = TransferFunction(0, 1, s)
        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
        >>> sys2 = TransferFunctionMatrix([[tf3, tf5], [tf5, tf5]])
        >>> F_1 = MIMOFeedback(sys1, sys2, 1)
        >>> pprint(F_1, use_unicode=False)
        /    [  s      1  ]    [5  0]   \\-1   [  s      1  ]
        |    [-----    -  ]    [-  -]   |     [-----    -  ]
        |    [1 - s    s  ]    [1  1]   |     [1 - s    s  ]
        |I - [            ]   *[    ]   |   * [            ]
        |    [  5    s - 1]    [0  0]   |     [  5    s - 1]
        |    [  -    -----]    [-  -]   |     [  -    -----]
        \\    [  1      s  ]{t} [1  1]{t}/     [  1      s  ]{t}
        >>> pprint(F_1.doit(), use_unicode=False)
        [  -s           s - 1       ]
        [-------     -----------    ]
        [6*s - 1     s*(6*s - 1)    ]
        [                           ]
        [5*s - 5  (s - 1)*(6*s + 24)]
        [-------  ------------------]
        [6*s - 1     s*(6*s - 1)    ]{t}

        If the user wants the resultant ``TransferFunctionMatrix`` object without
        canceling the common factors then the ``cancel`` kwarg should be passed ``False``.

        >>> pprint(F_1.doit(cancel=False), use_unicode=False)
        [           25*s*(1 - s)                          25 - 25*s              ]
        [       --------------------                    --------------           ]
        [       25*(1 - 6*s)*(1 - s)                    25*s*(1 - 6*s)           ]
        [                                                                        ]
        [s*(25*s - 25) + 5*(1 - s)*(6*s - 1)  s*(s - 1)*(6*s - 1) + s*(25*s - 25)]
        [-----------------------------------  -----------------------------------]
        [         (1 - s)*(6*s - 1)                        2                     ]
        [                                                 s *(6*s - 1)           ]{t}

        If the user wants the expanded form of the resultant transfer function matrix,
        the ``expand`` kwarg should be passed as ``True``.

        >>> pprint(F_1.doit(expand=True), use_unicode=False)
        [  -s          s - 1      ]
        [-------      --------    ]
        [6*s - 1         2        ]
        [             6*s  - s    ]
        [                         ]
        [            2            ]
        [5*s - 5  6*s  + 18*s - 24]
        [-------  ----------------]
        [6*s - 1         2        ]
        [             6*s  - s    ]{t}

        """
        _mat = self.sensitivity * self.sys1.doit()._expr_mat
        _resultant_tfm = _to_TFM(_mat, self.var)
        if cancel:
            _resultant_tfm = _resultant_tfm.simplify()
        if expand:
            _resultant_tfm = _resultant_tfm.expand()
        return _resultant_tfm

    def _eval_rewrite_as_TransferFunctionMatrix(self, sys1, sys2, sign, **kwargs):
        return self.doit()

    def __neg__(self):
        return MIMOFeedback(-self.sys1, -self.sys2, self.sign)