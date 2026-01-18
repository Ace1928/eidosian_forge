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
def test_transport_creation(self):
    from breezy.transport.fakevfat import FakeVFATTransportDecorator
    t = self.get_vfat_transport('.')
    self.assertIsInstance(t, FakeVFATTransportDecorator)