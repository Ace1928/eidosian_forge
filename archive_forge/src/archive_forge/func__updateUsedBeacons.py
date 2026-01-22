import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _updateUsedBeacons(self, beaconInformation):
    """
        Searches the adapter state and sentence data for information about
        which beacons where used, then adds it to the provided beacon
        information object.

        If no new beacon usage information is available, does nothing.

        @param beaconInformation: The beacon information object that beacon
            usage information will be added to (if necessary).
        @type beaconInformation: L{twisted.positioning.base.BeaconInformation}
        """
    for source in [self._state, self._sentenceData]:
        usedPRNs = source.get('_usedPRNs')
        if usedPRNs is not None:
            break
    else:
        return
    for beacon in beaconInformation.seenBeacons:
        if beacon.identifier in usedPRNs:
            beaconInformation.usedBeacons.add(beacon)