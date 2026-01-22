from distutils import errors
import imp
import os
import re
import shlex
import sys
import traceback
from setuptools.command import test
def ValidateGoogleTestDir(unused_dist, unused_attr, value):
    """Validate that the test directory is a directory."""
    if not os.path.isdir(value):
        raise errors.DistutilsSetupError('%s is not a directory' % value)