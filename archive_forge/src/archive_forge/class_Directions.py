from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
class Directions(Names):
    """
    The four cardinal directions (north, east, south, west).
    """
    NORTH = NamedConstant()
    EAST = NamedConstant()
    SOUTH = NamedConstant()
    WEST = NamedConstant()