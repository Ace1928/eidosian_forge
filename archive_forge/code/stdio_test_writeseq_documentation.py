import sys
from twisted.internet import protocol, stdio
from twisted.python import reflect

Main program for the child process run by
L{twisted.test.test_stdio.StandardInputOutputTests.test_writeSequence} to test
that ITransport.writeSequence() works for process transports.
