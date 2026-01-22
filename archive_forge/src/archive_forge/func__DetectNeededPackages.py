from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import subprocess
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import ext_runtime_adapter
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _DetectNeededPackages(gems):
    """Determines additional apt-get packages required by the given gems.

  Args:
    gems: ([str, ...]) A list of gems used by this application.

  Returns:
    ([str, ...]) A sorted list of strings indicating packages to install
  """
    package_set = set()
    for gem in gems:
        if gem in GEM_PACKAGES:
            package_set.update(GEM_PACKAGES[gem])
    packages = list(package_set)
    packages.sort()
    return packages