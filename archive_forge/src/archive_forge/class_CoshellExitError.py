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
class CoshellExitError(Exception):
    """The coshell exited."""

    def __init__(self, message, status=None):
        super(CoshellExitError, self).__init__(message)
        self.status = status