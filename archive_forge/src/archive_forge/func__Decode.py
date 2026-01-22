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
def _Decode(self, data):
    """Decodes external data if needed and returns internal string."""
    try:
        return data.decode(self._encoding)
    except (AttributeError, UnicodeError):
        return data