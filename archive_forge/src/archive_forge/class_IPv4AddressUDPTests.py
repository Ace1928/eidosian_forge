import os
import socket
from unittest import skipIf
from twisted.internet.address import (
from twisted.python.compat import nativeString
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
class IPv4AddressUDPTests(SynchronousTestCase, IPv4AddressTestCaseMixin):

    def buildAddress(self):
        """
        Create an arbitrary new L{IPv4Address} instance with a C{"UDP"}
        type.  A new instance is created for each call, but always for the
        same address.
        """
        return IPv4Address('UDP', '127.0.0.1', 0)

    def buildDifferentAddress(self):
        """
        Like L{buildAddress}, but with a different fixed address.
        """
        return IPv4Address('UDP', '127.0.0.2', 0)