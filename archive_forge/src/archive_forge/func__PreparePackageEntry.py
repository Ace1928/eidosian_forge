from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import posixpath
import re
import textwrap
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import utils
from googlecloudsdk.core import log
from six.moves import shlex_quote
def _PreparePackageEntry(package):
    """Returns the Dockerfile entries required to append at the end before entrypoint.

  Including:
  - copy the parent directory of the main executable into a docker container.
  - inject an entrypoint that executes a script or python module inside that
    directory.

  Args:
    package: (Package) Represents the main application copied to and run in the
      container.
  """
    parent_dir = os.path.dirname(package.script) or '.'
    copy_code = _GenerateCopyCommand(parent_dir, parent_dir, comment='Copy the source directory into the docker container.')
    return '\n{}\n'.format(copy_code)