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
class CreateJava17YamlCommand(_Command):
    """A command that creates a java17 runtime app.yaml from a jar file."""

    def EnsureInstalled(self):
        pass

    def GetPath(self):
        return None

    def GetArgs(self, descriptor, app_dir, staging_dir, explicit_appyaml=None):
        return None

    def Run(self, staging_area, descriptor, app_dir, explicit_appyaml=None):
        shutil.copy2(descriptor, staging_area)
        if explicit_appyaml:
            shutil.copyfile(explicit_appyaml, os.path.join(staging_area, 'app.yaml'))
        else:
            files.WriteFileContents(os.path.join(staging_area, 'app.yaml'), 'runtime: java17\ninstance_class: F2\n', private=True)
        manifest = jarfile.ReadManifest(descriptor)
        if manifest:
            main_entry = manifest.main_section.get('Main-Class')
            if main_entry is None:
                raise NoMainClassError()
            classpath_entry = manifest.main_section.get('Class-Path')
            if classpath_entry:
                libs = classpath_entry.split()
                for lib in libs:
                    dependent_file = os.path.join(app_dir, lib)
                    if os.path.isfile(dependent_file):
                        destination = os.path.join(staging_area, lib)
                        files.MakeDir(os.path.abspath(os.path.join(destination, os.pardir)))
                        try:
                            os.symlink(dependent_file, destination)
                        except (AttributeError, OSError):
                            log.debug('Could not symlink files in staging directory, falling back to copying')
                            shutil.copy(dependent_file, destination)
        return staging_area

    def __eq__(self, other):
        return isinstance(other, CreateJava17YamlCommand)