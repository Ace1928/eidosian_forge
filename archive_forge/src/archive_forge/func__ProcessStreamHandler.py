from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import errno
import os
import re
import signal
import subprocess
import sys
import threading
import time
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import parallel
from googlecloudsdk.core.util import platforms
import six
from six.moves import map
def _ProcessStreamHandler(proc, err=False, handler=log.Print):
    """Process output stream from a running subprocess in realtime."""
    stream = proc.stderr if err else proc.stdout
    stream_reader = stream.readline
    while True:
        line = stream_reader() or b''
        if not line and proc.poll() is not None:
            try:
                stream.close()
            except OSError:
                pass
            break
        line_str = line.decode('utf-8')
        line_str = line_str.rstrip('\r\n')
        if line_str:
            handler(line_str)