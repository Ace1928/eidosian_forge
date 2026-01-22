from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import os
import re
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app.images import config as images_config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
class GoConfigurator(ext_runtime.Configurator):
    """Generates configuration for a Go app."""

    def __init__(self, path, params):
        """Constructor.

    Args:
      path: (str) Root path of the source tree.
      params: (ext_runtime.Params) Parameters passed through to the
        fingerprinters.
    """
        self.root = path
        self.params = params

    def GetAllConfigFiles(self):
        all_config_files = []
        if not self.params.appinfo:
            app_yaml_path = os.path.join(self.root, 'app.yaml')
            if not os.path.exists(app_yaml_path):
                runtime = 'custom' if self.params.custom else 'go'
                app_yaml_contents = GO_APP_YAML.format(runtime=runtime)
                app_yaml = ext_runtime.GeneratedFile('app.yaml', app_yaml_contents)
                all_config_files.append(app_yaml)
        if self.params.custom or self.params.deploy:
            dockerfile_path = os.path.join(self.root, images_config.DOCKERFILE)
            if not os.path.exists(dockerfile_path):
                dockerfile = ext_runtime.GeneratedFile(images_config.DOCKERFILE, DOCKERFILE)
                all_config_files.append(dockerfile)
            dockerignore_path = os.path.join(self.root, '.dockerignore')
            if not os.path.exists(dockerignore_path):
                dockerignore = ext_runtime.GeneratedFile('.dockerignore', DOCKERIGNORE)
                all_config_files.append(dockerignore)
        return all_config_files

    def GenerateConfigs(self):
        """Generate config files for the module.

    Returns:
      (bool) True if files were created
    """
        if self.params.deploy:
            notify = log.info
        else:
            notify = log.status.Print
        cfg_files = self.GetAllConfigFiles()
        created = False
        for cfg_file in cfg_files:
            if cfg_file.WriteTo(self.root, notify):
                created = True
        if not created:
            notify('All config files already exist, not generating anything.')
        return created

    def GenerateConfigData(self):
        """Generate config files for the module.

    Returns:
      list(ext_runtime.GeneratedFile) list of generated files.
    """
        if self.params.deploy:
            notify = log.info
        else:
            notify = log.status.Print
        cfg_files = self.GetAllConfigFiles()
        for cfg_file in cfg_files:
            if cfg_file.filename == 'app.yaml':
                cfg_file.WriteTo(self.root, notify)
        final_cfg_files = []
        for f in cfg_files:
            if f.filename != 'app.yaml' and (not os.path.exists(os.path.join(self.root, f.filename))):
                final_cfg_files.append(f)
        return final_cfg_files