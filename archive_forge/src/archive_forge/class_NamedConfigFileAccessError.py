from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import re
import threading
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import properties_file
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
class NamedConfigFileAccessError(NamedConfigError):
    """Raise for errors dealing with file access errors."""

    def __init__(self, message, exc):
        super(NamedConfigFileAccessError, self).__init__('{0}.\n  {1}'.format(message, getattr(exc, 'strerror', exc)))