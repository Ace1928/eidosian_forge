import os
import sys
from zope.interface import implementer
from twisted.internet import interfaces
from twisted.python import log
from twisted.internet import protocol, reactor, stdio
A process that reads from stdin and out using Twisted.