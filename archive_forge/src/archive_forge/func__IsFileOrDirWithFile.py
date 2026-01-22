from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import os
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.ext import builtins
def _IsFileOrDirWithFile(path):
    """Determine if a path is a file or a directory with an appropriate file."""
    return os.path.isfile(path) or (os.path.isdir(path) and os.path.isfile(os.path.join(path, 'include.yaml')))