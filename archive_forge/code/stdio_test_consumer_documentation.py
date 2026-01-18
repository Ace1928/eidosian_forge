import sys
from twisted.internet import protocol, stdio
from twisted.protocols import basic
from twisted.python import log, reflect

Main program for the child process run by
L{twisted.test.test_stdio.StandardInputOutputTests.test_consumer} to test
that process transports implement IConsumer properly.
