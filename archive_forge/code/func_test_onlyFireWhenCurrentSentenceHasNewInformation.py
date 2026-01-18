from __future__ import annotations
import datetime
from operator import attrgetter
from typing import Callable, Iterable, TypedDict
from zope.interface import implementer
from constantly import NamedConstant
from typing_extensions import Literal, Protocol
from twisted.positioning import base, ipositioning, nmea
from twisted.positioning.base import Angles
from twisted.positioning.test.receiver import MockPositioningReceiver
from twisted.trial.unittest import TestCase
def test_onlyFireWhenCurrentSentenceHasNewInformation(self) -> None:
    """
        If the current sentence does not contain any new fields for a
        particular callback, that callback is not called; even if all
        necessary information is still in the state from one or more
        previous messages.
        """
    self.protocol.lineReceived(GPGGA)
    gpggaCallbacks = {'positionReceived', 'positionErrorReceived', 'altitudeReceived'}
    self.assertEqual(set(self.receiver.called.keys()), gpggaCallbacks)
    self.receiver.clear()
    self.assertNotEqual(self.adapter._state, {})
    self.protocol.lineReceived(GPHDT)
    gphdtCallbacks = {'headingReceived'}
    self.assertEqual(set(self.receiver.called.keys()), gphdtCallbacks)