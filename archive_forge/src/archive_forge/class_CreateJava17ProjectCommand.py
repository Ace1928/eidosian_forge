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
class CreateJava17ProjectCommand(_Command):
    """A command that creates a java17 runtime app.yaml."""

    def EnsureInstalled(self):
        pass

    def GetPath(self):
        return

    def GetArgs(self, descriptor, staging_dir, appyaml=None):
        return

    def Run(self, staging_area, descriptor, app_dir, explicit_appyaml=None):
        appenginewebxml = os.path.join(app_dir, 'src', 'main', 'webapp', 'WEB-INF', 'appengine-web.xml')
        if os.path.exists(appenginewebxml):
            raise self.error()
        if explicit_appyaml:
            shutil.copyfile(explicit_appyaml, os.path.join(staging_area, 'app.yaml'))
        else:
            appyaml = os.path.join(app_dir, 'src', 'main', 'appengine', 'app.yaml')
            if os.path.exists(appyaml):
                shutil.copy2(appyaml, staging_area)
            else:
                files.WriteFileContents(os.path.join(staging_area, 'app.yaml'), 'runtime: java17\ninstance_class: F2\n')
        for name in os.listdir(app_dir):
            if name == self.ignore:
                continue
            srcname = os.path.join(app_dir, name)
            dstname = os.path.join(staging_area, name)
            try:
                os.symlink(srcname, dstname)
            except (AttributeError, OSError):
                log.debug('Could not symlink files in staging directory, falling back to copying')
                if os.path.isdir(srcname):
                    files.CopyTree(srcname, dstname)
                else:
                    shutil.copy2(srcname, dstname)
        return staging_area

    def __eq__(self, other):
        return isinstance(other, CreateJava17ProjectCommand)