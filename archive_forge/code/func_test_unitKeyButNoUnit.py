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
def test_unitKeyButNoUnit(self) -> None:
    """
        Tests that if a unit key is provided but the unit isn't, the unit is
        automatically determined from the unit key.
        """

    class FakeSentence:
        """
            A fake sentence that just has "foo" and "fooUnits" attributes.
            """

        def __init__(self) -> None:
            self.foo = 1
            self.fooUnits = 'N'
    self.adapter.currentSentence = FakeSentence()
    self.adapter._fixUnits(unitKey='fooUnits')
    self.assertNotEqual(self.adapter._sentenceData['foo'], 1)