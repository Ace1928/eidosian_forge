from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
class InvalidDependencyError(errors.Error):
    """Raised on attempts to create an invalid dependency.

  Invalid dependencies are self-dependencies and those that involve nodes that
  do not exist.
  """