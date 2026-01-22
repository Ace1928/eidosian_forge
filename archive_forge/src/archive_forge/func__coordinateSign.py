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
def _coordinateSign(hemisphere: str) -> Literal[1, -1]:
    """
    Return the sign of a coordinate.

    This is C{1} if the coordinate is in the northern or eastern hemispheres,
    C{-1} otherwise.

    @param hemisphere: NMEA shorthand for the hemisphere. One of "NESW".
    @type hemisphere: C{str}

    @return: The sign of the coordinate value.
    @rtype: C{int}
    """
    return 1 if hemisphere in 'NE' else -1