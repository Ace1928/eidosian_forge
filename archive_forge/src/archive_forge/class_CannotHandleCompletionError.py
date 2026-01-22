from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shlex
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
class CannotHandleCompletionError(Exception):
    """Error for when completions cannot be handled."""
    pass