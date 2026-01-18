import random
from zope.interface import implementer
from twisted.internet import error, interfaces
from twisted.names import client, dns
from twisted.names.error import DNSNameError
from twisted.python.compat import nativeString
Disconnect whatever our are state is.