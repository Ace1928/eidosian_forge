from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import os.path
import signal
import subprocess
import sys
import threading
from googlecloudsdk.command_lib.code import json_stream
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
import six
def _FindSkaffold():
    """Find the path to the skaffold executable."""
    skaffold = properties.VALUES.code.skaffold_path_override.Get() or _FindOrInstallSkaffoldComponent() or file_utils.FindExecutableOnPath('skaffold')
    if not skaffold:
        raise EnvironmentError('Unable to locate skaffold.')
    return skaffold