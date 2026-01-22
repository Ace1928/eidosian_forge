import _thread
import errno
import io
import os
import sys
import time
import breezy
from ...lazy_import import lazy_import
import select
import socket
import weakref
from breezy import (
from breezy.i18n import gettext
from breezy.bzr.smart import client, protocol, request, signals, vfs
from breezy.transport import ssh
from ... import errors, osutils
class TooManyConcurrentRequests(errors.InternalBzrError):
    _fmt = "The medium '%(medium)s' has reached its concurrent request limit. Be sure to finish_writing and finish_reading on the currently open request."

    def __init__(self, medium):
        self.medium = medium