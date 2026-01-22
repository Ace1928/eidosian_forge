import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _statefulUpdate(self, sentenceKey):
    """
        Does a stateful update of a particular positioning attribute.
        Specifically, this will mutate an object in the current sentence data.

        Using the C{sentenceKey}, this will get a tuple containing, in order,
        the key name in the current state and sentence data, a factory for
        new values, the attribute to update, and a converter from sentence
        data (in NMEA notation) to something useful.

        If the sentence data doesn't have this data yet, it is grabbed from
        the state. If that doesn't have anything useful yet either, the
        factory is called to produce a new, empty object. Either way, the
        object ends up in the sentence data.

        @param sentenceKey: The name of the key in the sentence attributes,
            C{NMEAAdapter._STATEFUL_UPDATE} dictionary and the adapter state.
        @type sentenceKey: C{str}
        """
    key, factory, attr, converter = self._STATEFUL_UPDATE[sentenceKey]
    if key not in self._sentenceData:
        try:
            self._sentenceData[key] = self._state[key]
        except KeyError:
            self._sentenceData[key] = factory()
    newValue = converter(getattr(self.currentSentence, sentenceKey))
    setattr(self._sentenceData[key], attr, newValue)