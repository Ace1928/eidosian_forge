from .sage_helper import _within_sage
from .pari import *
import re
class SnappyNumbersMetaclass(ClasscallMetaclass):
    """
        Metaclass for Sage parents of SnapPy Number objects.
        """

    def __new__(mcs, name, bases, dict):
        dict['category'] = lambda self: Fields()
        return ClasscallMetaclass.__new__(mcs, name, bases, dict)