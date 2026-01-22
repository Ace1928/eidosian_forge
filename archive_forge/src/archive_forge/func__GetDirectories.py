from a man-ish style runtime document.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import os
import re
import shlex
import subprocess
import tarfile
import textwrap
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.static_completion import generate as generate_static
from googlecloudsdk.command_lib.static_completion import lookup
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
from six.moves import range
def _GetDirectories(directory=None, warn_on_exceptions=False):
    """Returns the list of directories to search for CLI trees.

  Args:
    directory: The directory containing the CLI tree JSON files. If None
      then the default installation and config directories are used.
    warn_on_exceptions: Emits warning messages in lieu of exceptions.
  """
    directories = []
    if directory:
        directories.append(directory)
    else:
        try:
            directories.append(cli_tree.CliTreeDir())
        except cli_tree.SdkRootNotFoundError as e:
            if not warn_on_exceptions:
                raise
            log.warning(six.text_type(e))
        directories.append(cli_tree.CliTreeConfigDir())
    return directories