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
def _invalidFixTest(self, sentenceData: dict[str, str]) -> None:
    """
        Sentences with an invalid fix or data mode result in empty
        state (ie, the data isn't used).
        """
    self._fixerTest(sentenceData, {})