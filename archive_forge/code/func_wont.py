import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
def wont(self, option):
    """
        Indicate we are not willing to enable an option.
        """
    s = self.getOptionState(option)
    if s.us.negotiating or s.him.negotiating:
        return defer.fail(AlreadyNegotiating(option))
    elif s.us.state == 'no':
        return defer.fail(AlreadyDisabled(option))
    else:
        s.us.negotiating = True
        s.us.onResult = d = defer.Deferred()
        self._wont(option)
        return d