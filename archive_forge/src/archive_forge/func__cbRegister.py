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
def _cbRegister(self, registration, message):
    response = self.responseFromRequest(200, message)
    if registration.contactURL != None:
        response.addHeader('contact', registration.contactURL.toString())
        response.addHeader('expires', '%d' % registration.secondsToExpiry)
    response.addHeader('content-length', '0')
    self.deliverResponse(response)