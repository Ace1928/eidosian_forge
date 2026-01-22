import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def _cleanupGlobalDeferreds(self):
    """
        All pending requests that have returned a deferred must be errbacked
        when this service is stopped, otherwise they might be left uncalled and
        uncallable.
        """
    for d in self.deferreds['global']:
        d.errback(error.ConchError('Connection stopped.'))
    del self.deferreds['global'][:]