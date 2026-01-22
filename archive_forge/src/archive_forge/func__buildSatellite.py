from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def _buildSatellite(**kw: float) -> base.Satellite:
    kwargs = dict(self.satelliteKwargs)
    kwargs.update(kw)
    return base.Satellite(**kwargs)