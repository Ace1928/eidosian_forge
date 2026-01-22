from within calliope.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
from functools import wraps
import os
import sys
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
import six
def _NonAsciiIndex(s):
    """Returns the index of the first non-ascii char in s, -1 if all ascii."""
    if isinstance(s, six.text_type):
        for i, c in enumerate(s):
            try:
                c.encode('ascii')
            except (AttributeError, UnicodeError):
                return i
    else:
        for i, b in enumerate(s):
            try:
                b.decode('ascii')
            except (AttributeError, UnicodeError):
                return i
    return -1