from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
class Satellite(PositioningBeacon):
    """
    A satellite.

    @ivar azimuth: The azimuth of the satellite. This is the heading (positive
        angle relative to true north) where the satellite appears to be to the
        device.
    @ivar elevation: The (positive) angle above the horizon where this
        satellite appears to be to the device.
    @ivar signalToNoiseRatio: The signal to noise ratio of the signal coming
        from this satellite.
    """

    def __init__(self, identifier, azimuth=None, elevation=None, signalToNoiseRatio=None):
        """
        Initializes a satellite object.

        @param identifier: The PRN (unique identifier) of this satellite.
        @type identifier: C{int}
        @param azimuth: The azimuth of the satellite (see instance variable
            documentation).
        @type azimuth: C{float}
        @param elevation: The elevation of the satellite (see instance variable
            documentation).
        @type elevation: C{float}
        @param signalToNoiseRatio: The signal to noise ratio of the connection
            to this satellite (see instance variable documentation).
        @type signalToNoiseRatio: C{float}
        """
        PositioningBeacon.__init__(self, int(identifier))
        self.azimuth = azimuth
        self.elevation = elevation
        self.signalToNoiseRatio = signalToNoiseRatio

    def __repr__(self) -> str:
        """
        Returns a string representation of this Satellite.

        @return: The string representation.
        @rtype: C{str}
        """
        template = '<Satellite ({s.identifier}), azimuth: {s.azimuth}, elevation: {s.elevation}, snr: {s.signalToNoiseRatio}>'
        return template.format(s=self)