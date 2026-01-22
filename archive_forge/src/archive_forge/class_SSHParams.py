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
class SSHParams:
    """A set of parameters for starting a remote bzr via SSH."""

    def __init__(self, host, port=None, username=None, password=None, bzr_remote_path='bzr'):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.bzr_remote_path = bzr_remote_path