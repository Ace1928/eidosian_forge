from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
class Coordinate(Angle):
    """
    A coordinate.

    @ivar angle: The value of the coordinate in decimal degrees, with the usual
        rules for sign (northern and eastern hemispheres are positive, southern
        and western hemispheres are negative).
    @type angle: C{float}
    """

    def __init__(self, angle, coordinateType=None):
        """
        Initializes a coordinate.

        @param angle: The angle of this coordinate in decimal degrees. The
            hemisphere is determined by the sign (north and east are positive).
            If this coordinate describes a latitude, this value must be within
            -90.0 and +90.0 (exclusive). If this value describes a longitude,
            this value must be within -180.0 and +180.0 (exclusive).
        @type angle: C{float}
        @param coordinateType: The coordinate type. One of L{Angles.LATITUDE},
            L{Angles.LONGITUDE} or L{None} if unknown.
        """
        if coordinateType not in [Angles.LATITUDE, Angles.LONGITUDE, None]:
            raise ValueError('coordinateType must be one of Angles.LATITUDE, Angles.LONGITUDE or None, was {!r}'.format(coordinateType))
        Angle.__init__(self, angle, coordinateType)

    @property
    def hemisphere(self):
        """
        Gets the hemisphere of this coordinate.

        @return: A symbolic constant representing a hemisphere (one of
            L{Angles})
        """
        if self.angleType is Angles.LATITUDE:
            if self.inDecimalDegrees < 0:
                return Directions.SOUTH
            else:
                return Directions.NORTH
        elif self.angleType is Angles.LONGITUDE:
            if self.inDecimalDegrees < 0:
                return Directions.WEST
            else:
                return Directions.EAST
        else:
            raise ValueError('unknown coordinate type (cant find hemisphere)')