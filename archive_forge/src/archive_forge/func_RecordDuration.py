from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import tempfile
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
@contextlib.contextmanager
def RecordDuration(span_name):
    """Record duration of a span of time.

  Two timestamps will be sent, and the duration in between will be considered as
  the client side latency of this span.

  Args:
    span_name: str, The name of the span to time.

  Yields:
    None
  """
    CustomTimedEvent(span_name + '_start')
    yield
    CustomTimedEvent(span_name)