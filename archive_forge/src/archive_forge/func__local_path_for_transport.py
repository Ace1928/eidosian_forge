import errno
import os.path
import socket
import sys
import threading
import time
from ... import errors, trace
from ... import transport as _mod_transport
from ...hooks import Hooks
from ...i18n import gettext
from ...lazy_import import lazy_import
from breezy.bzr.smart import (
from breezy.transport import (
from breezy import (
def _local_path_for_transport(transport):
    """Return a local path for transport, if reasonably possible.

    This function works even if transport's url has a "readonly+" prefix,
    unlike local_path_from_url.

    This essentially recovers the --directory argument the user passed to "bzr
    serve" from the transport passed to serve_bzr.
    """
    try:
        base_url = transport.external_url()
    except (errors.InProcessTransport, NotImplementedError):
        return None
    else:
        if base_url.startswith('readonly+'):
            base_url = base_url[len('readonly+'):]
        try:
            return urlutils.local_path_from_url(base_url)
        except urlutils.InvalidURL:
            return None