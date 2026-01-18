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
def unauthorized(self, message, host, port):
    m = self.responseFromRequest(401, message)
    for scheme, auth in self.authorizers.items():
        chal = auth.getChallenge((host, port))
        if chal is None:
            value = f'{scheme.title()} realm="{self.host}"'
        else:
            value = f'{scheme.title()} {chal},realm="{self.host}"'
        m.headers.setdefault('www-authenticate', []).append(value)
    self.deliverResponse(m)