import os
import socket
from unittest import skipIf
from twisted.internet.address import (
from twisted.python.compat import nativeString
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
@skipIf(unixSkip, "platform doesn't support UNIX sockets.")
class EmptyUNIXAddressTests(SynchronousTestCase, AddressTestCaseMixin):
    """
    Tests for L{UNIXAddress} operations involving a L{None} address.
    """
    addressArgSpec = (('name', '%r'),)

    def setUp(self):
        self._socketAddress = self.mktemp()

    def buildAddress(self):
        """
        Create an arbitrary new L{UNIXAddress} instance.  A new instance is
        created for each call, but always for the same address. This builds it
        with a fixed address of L{None}.
        """
        return UNIXAddress(None)

    def buildDifferentAddress(self):
        """
        Like L{buildAddress}, but with a random temporary directory.
        """
        return UNIXAddress(self._socketAddress)

    @skipIf(symlinkSkip, 'Platform does not support symlinks')
    def test_comparisonOfLinkedFiles(self):
        """
        A UNIXAddress referring to a L{None} address does not
        compare equal to a UNIXAddress referring to a symlink.
        """
        linkName = self.mktemp()
        with open(self._socketAddress, 'w') as self.fd:
            os.symlink(os.path.abspath(self._socketAddress), linkName)
            self.assertNotEqual(UNIXAddress(self._socketAddress), UNIXAddress(None))
            self.assertNotEqual(UNIXAddress(None), UNIXAddress(self._socketAddress))

    def test_emptyHash(self):
        """
        C{__hash__} can be used to get a hash of an address, even one referring
        to L{None} rather than a real path.
        """
        addr = self.buildAddress()
        d = {addr: True}
        self.assertTrue(d[self.buildAddress()])