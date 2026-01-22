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
class StageAppWithoutAppYamlCommand(_Command):
    """A command that creates a staged directory with an optional app.yaml."""

    def EnsureInstalled(self):
        pass

    def GetPath(self):
        return None

    def GetArgs(self, descriptor, app_dir, staging_dir, explicit_appyaml=None):
        return None

    def Run(self, staging_area, descriptor, app_dir, explicit_appyaml=None):
        scratch_area = os.path.join(staging_area, 'scratch')
        if os.path.isdir(app_dir):
            files.CopyTree(app_dir, scratch_area)
        else:
            os.mkdir(scratch_area)
            shutil.copy2(app_dir, scratch_area)
        if explicit_appyaml:
            shutil.copyfile(explicit_appyaml, os.path.join(scratch_area, 'app.yaml'))
        return scratch_area

    def __eq__(self, other):
        return isinstance(other, StageAppWithoutAppYamlCommand)