from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
class Climb(_BaseSpeed):
    """
    The climb ("vertical speed") of an object.
    """

    def __init__(self, climb):
        """
        Initializes a L{Climb} object.

        @param climb: The climb that this object represents, expressed in
            meters per second.
        @type climb: C{float}
        """
        _BaseSpeed.__init__(self, climb)