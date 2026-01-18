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
def test_positionErrorUpdateAcrossStates(self) -> None:
    """
        The positioning error is updated across multiple states.
        """
    sentences = [GPGSA] + GPGSV_SEQ
    callbacksFired = ['positionErrorReceived', 'beaconInformationReceived']

    def _getIdentifiers(beacons: Iterable[base.Satellite]) -> list[int]:
        return sorted(map(attrgetter('identifier'), beacons))

    def checkBeaconInformation() -> None:
        beaconInformation = self.adapter._state['beaconInformation']
        seenIdentifiers = _getIdentifiers(beaconInformation.seenBeacons)
        expected = [3, 4, 6, 13, 14, 16, 18, 19, 22, 24, 27]
        self.assertEqual(seenIdentifiers, expected)
        usedIdentifiers = _getIdentifiers(beaconInformation.usedBeacons)
        self.assertEqual(usedIdentifiers, [14, 18, 19, 22, 27])
    self._receiverTest(sentences, callbacksFired, checkBeaconInformation)