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
def test_url_preserves_pathfiltering(self):
    """Calling get_transport on a pathfiltered transport's base should
        produce a transport with exactly the same behaviour as the original
        pathfiltered transport.

        This is so that it is not possible to escape (accidentally or
        otherwise) the filtering by doing::
            url = filtered_transport.base
            parent_url = urlutils.join(url, '..')
            new_t = transport.get_transport_from_url(parent_url)
        """
    t = self.make_pf_transport()
    new_t = transport.get_transport_from_url(t.base)
    self.assertEqual(t.server, new_t.server)
    self.assertEqual(t.base, new_t.base)