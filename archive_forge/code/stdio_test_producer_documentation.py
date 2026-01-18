import sys
from twisted.internet import protocol, stdio
from twisted.python import log, reflect

Main program for the child process run by
L{twisted.test.test_stdio.StandardInputOutputTests.test_producer} to test
that process transports implement IProducer properly.
