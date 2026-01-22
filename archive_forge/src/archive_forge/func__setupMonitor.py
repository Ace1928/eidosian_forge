import os
import warnings
from zope.interface import implementer
from twisted.application import internet, service
from twisted.cred.portal import Portal
from twisted.internet import defer
from twisted.mail import protocols, smtp
from twisted.mail.interfaces import IAliasableDomain, IDomain
from twisted.python import log, util
def _setupMonitor(self):
    """
        Schedule the next monitoring call.
        """
    from twisted.internet import reactor
    t, self.index = self.intervals.next()
    self._call = reactor.callLater(t, self._monitor)