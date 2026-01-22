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
def addHeader(self, name, value):
    name = name.lower()
    name = longHeaders.get(name, name)
    if name == 'content-length':
        self.length = int(value)
    self.headers.setdefault(name, []).append(value)