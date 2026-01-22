from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
def UpcomingSupportedVersionMessage(self):
    return 'Please use Python version {0}.{1} and up.'.format(PythonVersion.UPCOMING_PY3_MIN_SUPPORTED_VERSION[0], PythonVersion.UPCOMING_PY3_MIN_SUPPORTED_VERSION[1])