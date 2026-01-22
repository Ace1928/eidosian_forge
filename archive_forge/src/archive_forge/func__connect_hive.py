from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
import collections
import warnings
def _connect_hive(self, hive):
    """
        Connect to the specified hive of a remote Registry.

        @note: The connection will be cached, to close all connections and
            erase this cache call the L{close} method.

        @type  hive: int
        @param hive: Hive to connect to.

        @rtype:  L{win32.RegistryKeyHandle}
        @return: Open handle to the remote Registry hive.
        """
    try:
        handle = self._remote_hives[hive]
    except KeyError:
        handle = win32.RegConnectRegistry(self._machine, hive)
        self._remote_hives[hive] = handle
    return handle