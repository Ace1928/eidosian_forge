import os
import signal
import struct
import sys
from zope.interface import implementer
from twisted.conch.interfaces import (
from twisted.conch.ssh import channel, common, connection
from twisted.internet import interfaces, protocol
from twisted.logger import Logger
from twisted.python.compat import networkString
def _getSignalName(self, signum):
    """
        Get a signal name given a signal number.
        """
    if self._signalValuesToNames is None:
        self._signalValuesToNames = {}
        for signame in SUPPORTED_SIGNALS:
            signame = 'SIG' + signame
            sigvalue = getattr(signal, signame, None)
            if sigvalue is not None:
                self._signalValuesToNames[sigvalue] = signame
        for k, v in signal.__dict__.items():
            if k.startswith('SIG') and (not k.startswith('SIG_')):
                if v not in self._signalValuesToNames:
                    self._signalValuesToNames[v] = k + '@' + sys.platform
    return self._signalValuesToNames[signum]