from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
class CoordinateSym(Symbol):
    """
    A coordinate symbol/base scalar associated wrt a Reference Frame.

    Ideally, users should not instantiate this class. Instances of
    this class must only be accessed through the corresponding frame
    as 'frame[index]'.

    CoordinateSyms having the same frame and index parameters are equal
    (even though they may be instantiated separately).

    Parameters
    ==========

    name : string
        The display name of the CoordinateSym

    frame : ReferenceFrame
        The reference frame this base scalar belongs to

    index : 0, 1 or 2
        The index of the dimension denoted by this coordinate variable

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, CoordinateSym
    >>> A = ReferenceFrame('A')
    >>> A[1]
    A_y
    >>> type(A[0])
    <class 'sympy.physics.vector.frame.CoordinateSym'>
    >>> a_y = CoordinateSym('a_y', A, 1)
    >>> a_y == A[1]
    True

    """

    def __new__(cls, name, frame, index):
        assumptions = {}
        super()._sanitize(assumptions, cls)
        obj = super().__xnew__(cls, name, **assumptions)
        _check_frame(frame)
        if index not in range(0, 3):
            raise ValueError('Invalid index specified')
        obj._id = (frame, index)
        return obj

    @property
    def frame(self):
        return self._id[0]

    def __eq__(self, other):
        if isinstance(other, CoordinateSym):
            if other._id == self._id:
                return True
        return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return (self._id[0].__hash__(), self._id[1]).__hash__()