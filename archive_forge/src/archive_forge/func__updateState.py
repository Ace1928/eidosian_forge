import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _updateState(self):
    """
        Updates the current state with the new information from the sentence.
        """
    self._updateBeaconInformation()
    self._combineDateAndTime()
    self._state.update(self._sentenceData)