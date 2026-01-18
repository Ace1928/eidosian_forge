import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
def will_no_false(self, state, option):
    if self.enableRemote(option):
        state.him.state = 'yes'
        self._do(option)
    else:
        self._dont(option)