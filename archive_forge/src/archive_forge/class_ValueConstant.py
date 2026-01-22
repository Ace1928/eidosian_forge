from __future__ import division, absolute_import
from functools import partial
from itertools import count
from operator import and_, or_, xor
class ValueConstant(_Constant):
    """
    L{ValueConstant} defines an attribute to be a named constant within a
    collection defined by a L{Values} subclass.

    L{ValueConstant} is only for use in the definition of L{Values} subclasses.
    Do not instantiate L{ValueConstant} elsewhere and do not subclass it.
    """

    def __init__(self, value):
        _Constant.__init__(self)
        self.value = value