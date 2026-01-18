import errno
import sys
import time
from array import array
from socket import AF_INET, AF_INET6, SOCK_STREAM, SOL_SOCKET, socket
from struct import pack
from unittest import skipIf
from zope.interface.verify import verifyClass
from twisted.internet.interfaces import IPushProducer
from twisted.python.log import msg
from twisted.trial.unittest import TestCase

        Verify that we don't lose an event when more than EVENTS_PER_LOOP
        events occur in the same reactor iteration
        