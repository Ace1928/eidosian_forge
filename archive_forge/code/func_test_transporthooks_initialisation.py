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
def test_transporthooks_initialisation(self):
    """Check all expected transport hook points are set up"""
    hookpoint = transport.TransportHooks()
    self.assertTrue('post_connect' in hookpoint, 'post_connect not in {}'.format(hookpoint))