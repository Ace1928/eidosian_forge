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
def _coordinateType(hemisphere: str) -> NamedConstant:
    """
    Return the type of a coordinate.

    This is L{Angles.LATITUDE} if the coordinate is in the northern or
    southern hemispheres, L{Angles.LONGITUDE} otherwise.

    @param hemisphere: NMEA shorthand for the hemisphere. One of "NESW".
    @type hemisphere: C{str}

    @return: The type of the coordinate (L{Angles.LATITUDE} or
        L{Angles.LONGITUDE})
    """
    return Angles.LATITUDE if hemisphere in 'NS' else Angles.LONGITUDE