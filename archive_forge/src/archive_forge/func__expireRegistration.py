import socket
import time
import warnings
from collections import OrderedDict
from typing import Dict, List
from zope.interface import Interface, implementer
from twisted import cred
from twisted.internet import defer, protocol, reactor
from twisted.protocols import basic
from twisted.python import log
def _expireRegistration(self, username):
    try:
        dc, url = self.users[username]
    except KeyError:
        return defer.fail(LookupError('no such user'))
    else:
        dc.cancel()
        del self.users[username]
    return defer.succeed(Registration(0, url))