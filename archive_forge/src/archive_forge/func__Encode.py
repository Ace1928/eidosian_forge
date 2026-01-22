from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import locale
import os
import re
import signal
import subprocess
from googlecloudsdk.core.util import encoding
import six
def _Encode(self, string):
    """Encodes internal string if needed and returns external data."""
    try:
        return string.encode(self._encoding)
    except UnicodeError:
        return string