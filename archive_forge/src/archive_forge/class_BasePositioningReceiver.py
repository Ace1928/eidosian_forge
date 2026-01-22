from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
@implementer(ipositioning.IPositioningReceiver)
class BasePositioningReceiver:
    """
    A base positioning receiver.

    This class would be a good base class for building positioning
    receivers. It implements the interface (so you don't have to) with stub
    methods.

    People who want to implement positioning receivers should subclass this
    class and override the specific callbacks they want to handle.
    """

    def timeReceived(self, time):
        """
        Implements L{IPositioningReceiver.timeReceived} stub.
        """

    def headingReceived(self, heading):
        """
        Implements L{IPositioningReceiver.headingReceived} stub.
        """

    def speedReceived(self, speed):
        """
        Implements L{IPositioningReceiver.speedReceived} stub.
        """

    def climbReceived(self, climb):
        """
        Implements L{IPositioningReceiver.climbReceived} stub.
        """

    def positionReceived(self, latitude, longitude):
        """
        Implements L{IPositioningReceiver.positionReceived} stub.
        """

    def positionErrorReceived(self, positionError):
        """
        Implements L{IPositioningReceiver.positionErrorReceived} stub.
        """

    def altitudeReceived(self, altitude):
        """
        Implements L{IPositioningReceiver.altitudeReceived} stub.
        """

    def beaconInformationReceived(self, beaconInformation):
        """
        Implements L{IPositioningReceiver.beaconInformationReceived} stub.
        """