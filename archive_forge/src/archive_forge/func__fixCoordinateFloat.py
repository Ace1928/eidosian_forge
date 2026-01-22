import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _fixCoordinateFloat(self, coordinateType):
    """
        Turns the NMEAProtocol coordinate format into Python float.

        @param coordinateType: The coordinate type.
        @type coordinateType: One of L{Angles.LATITUDE} or L{Angles.LONGITUDE}.
        """
    if coordinateType is Angles.LATITUDE:
        coordinateName = 'latitude'
    else:
        coordinateName = 'longitude'
    nmeaCoordinate = getattr(self.currentSentence, coordinateName + 'Float')
    left, right = nmeaCoordinate.split('.')
    degrees, minutes = (int(left[:-2]), float(f'{left[-2:]}.{right}'))
    angle = degrees + minutes / 60
    coordinate = base.Coordinate(angle, coordinateType)
    self._sentenceData[coordinateName] = coordinate