import random
from typing import Any, Callable, Optional
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.internet.interfaces import IAddress, ITransport
from twisted.logger import _loggerFor
from twisted.python import components, failure, log
class ServerFactory(Factory):
    """
    Subclass this to indicate that your protocol.Factory is only usable for servers.
    """