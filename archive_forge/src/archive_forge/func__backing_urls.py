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
def _backing_urls(self):
    urls = [self.backing_transport.base]
    try:
        urls.append(self.backing_transport.external_url())
    except errors.InProcessTransport:
        pass
    return urls