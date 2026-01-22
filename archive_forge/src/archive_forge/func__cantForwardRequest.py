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
def _cantForwardRequest(self, error, message):
    error.trap(LookupError)
    del message.headers['via'][0]
    self.deliverResponse(self.responseFromRequest(404, message))