from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import os
import re
import shutil
import tempfile
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import runtime_registry
from googlecloudsdk.command_lib.app import jarfile
from googlecloudsdk.command_lib.util import java
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
class NoopCommand(_Command):
    """A command that does nothing.

  Many runtimes do not require a staging step; this isn't a problem.
  """

    def EnsureInstalled(self):
        pass

    def GetPath(self):
        return None

    def GetArgs(self, descriptor, app_dir, staging_dir, explicit_appyaml=None):
        return None

    def Run(self, staging_area, descriptor, app_dir, explicit_appyaml=None):
        """Does nothing."""
        pass

    def __eq__(self, other):
        return isinstance(other, NoopCommand)