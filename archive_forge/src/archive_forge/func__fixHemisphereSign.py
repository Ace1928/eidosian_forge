import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _fixHemisphereSign(self, coordinateType, sentenceDataKey=None):
    """
        Fixes the sign for a hemisphere.

        This method must be called after the magnitude for the thing it
        determines the sign of has been set. This is done by the following
        functions:

            - C{self.FIXERS['magneticVariation']}
            - C{self.FIXERS['latitudeFloat']}
            - C{self.FIXERS['longitudeFloat']}

        @param coordinateType: Coordinate type. One of L{Angles.LATITUDE},
            L{Angles.LONGITUDE} or L{Angles.VARIATION}.
        @param sentenceDataKey: The key name of the hemisphere sign being
            fixed in the sentence data. If unspecified, C{coordinateType} is
            used.
        @type sentenceDataKey: C{str} (unless L{None})
        """
    sentenceDataKey = sentenceDataKey or coordinateType
    sign = self._getHemisphereSign(coordinateType)
    self._sentenceData[sentenceDataKey].setSign(sign)