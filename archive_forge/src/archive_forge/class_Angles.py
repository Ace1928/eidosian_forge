from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
class Angles(Names):
    """
    The types of angles.

    @cvar LATITUDE: Angle representing a latitude of an object.
    @type LATITUDE: L{NamedConstant}

    @cvar LONGITUDE: Angle representing the longitude of an object.
    @type LONGITUDE: L{NamedConstant}

    @cvar HEADING: Angle representing the heading of an object.
    @type HEADING: L{NamedConstant}

    @cvar VARIATION: Angle representing a magnetic variation.
    @type VARIATION: L{NamedConstant}

    """
    LATITUDE = NamedConstant()
    LONGITUDE = NamedConstant()
    HEADING = NamedConstant()
    VARIATION = NamedConstant()