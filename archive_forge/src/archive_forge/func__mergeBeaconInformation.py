import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _mergeBeaconInformation(self, newBeaconInformation):
    """
        Merges beacon information in the adapter state (if it exists) into
        the provided beacon information. Specifically, this merges used and
        seen beacons.

        If the adapter state has no beacon information, does nothing.

        @param newBeaconInformation: The beacon information object that beacon
            information will be merged into (if necessary).
        @type newBeaconInformation: L{twisted.positioning.base.BeaconInformation}
        """
    old = self._state.get('_partialBeaconInformation')
    if old is None:
        return
    for attr in ['seenBeacons', 'usedBeacons']:
        getattr(newBeaconInformation, attr).update(getattr(old, attr))