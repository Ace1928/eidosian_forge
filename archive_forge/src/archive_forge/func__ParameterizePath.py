from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
def _ParameterizePath(path):
    """Return path with $HOME prefix replaced by ~."""
    home = files.GetHomeDir() + os.path.sep
    if path.startswith(home):
        return '~' + os.path.sep + path[len(home):]
    return path