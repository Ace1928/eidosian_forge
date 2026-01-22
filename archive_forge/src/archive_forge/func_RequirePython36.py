from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
def RequirePython36(cmd='gcloud'):
    """Verifies that the python version is 3.6+.

  Args:
    cmd: The string command that requires python 3.6+.

  Raises:
    InvalidPythonVersion: if the python version is not 3.6+.
  """
    if sys.version_info.major < 3 or (sys.version_info.major == 3 and sys.version_info.minor < 6):
        raise InvalidPythonVersion('The `{}` command requires python 3.6 or greater. Please update the\n        python version to use this command.'.format(cmd))