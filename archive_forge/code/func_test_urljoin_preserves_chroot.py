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
def test_urljoin_preserves_chroot(self):
    """Using urlutils.join(url, '..') on a chroot URL should not produce a
        URL that escapes the intended chroot.

        This is so that it is not possible to escape a chroot by doing::
            url = chroot_transport.base
            parent_url = urlutils.join(url, '..')
            new_t = transport.get_transport_from_url(parent_url)
        """
    server = chroot.ChrootServer(transport.get_transport_from_url('memory:///path/'))
    self.start_server(server)
    t = transport.get_transport_from_url(server.get_url())
    self.assertRaises(urlutils.InvalidURLJoin, urlutils.join, t.base, '..')