import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _updateBeaconInformation(self):
    """
        Updates existing beacon information state with new data.
        """
    new = self._sentenceData.get('_partialBeaconInformation')
    if new is None:
        return
    self._updateUsedBeacons(new)
    self._mergeBeaconInformation(new)
    if self.currentSentence._isLastGSVSentence():
        if not self.currentSentence._isFirstGSVSentence():
            del self._state['_partialBeaconInformation']
        bi = self._sentenceData.pop('_partialBeaconInformation')
        self._sentenceData['beaconInformation'] = bi