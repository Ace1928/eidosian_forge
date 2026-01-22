import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
class TestTransportImplementation(tests.TestCaseInTempDir):
    """Implementation verification for transports.

    To verify a transport we need a server factory, which is a callable
    that accepts no parameters and returns an implementation of
    breezy.transport.Server.

    That Server is then used to construct transport instances and test
    the transport via loopback activity.

    Currently this assumes that the Transport object is connected to the
    current working directory.  So that whatever is done
    through the transport, should show up in the working
    directory, and vice-versa. This is a bug, because its possible to have
    URL schemes which provide access to something that may not be
    result in storage on the local disk, i.e. due to file system limits, or
    due to it being a database or some other non-filesystem tool.

    This also tests to make sure that the functions work with both
    generators and lists (assuming iter(list) is effectively a generator)
    """

    def setUp(self):
        super().setUp()
        self._server = self.transport_server()
        self.start_server(self._server)

    def get_transport(self, relpath=None):
        """Return a connected transport to the local directory.

        :param relpath: a path relative to the base url.
        """
        base_url = self._server.get_url()
        url = self._adjust_url(base_url, relpath)
        t = transport.get_transport_from_url(url)
        if not isinstance(t, self.transport_class):
            t = self.transport_class(url)
        return t