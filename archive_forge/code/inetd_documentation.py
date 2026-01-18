import os
from twisted.internet import fdesc, process, reactor
from twisted.internet.protocol import Protocol, ServerFactory
from twisted.protocols import wire
Forks a child process on connectionMade, passing the socket as fd 0.