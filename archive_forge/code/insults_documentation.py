from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString

        Parse the given data from a terminal server, dispatching to event
        handlers defined by C{self.terminal}.
        