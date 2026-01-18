import errno
from select import (
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.python import log
Poll the poller for new events.