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
def test_raiseOnUnknownSentenceType(self) -> None:
    """
        Receiving a well-formed sentence of unknown type raises
        C{ValueError}.
        """
    self.assertRaisesOnSentence(ValueError, b'$GPBOGUS*5b')