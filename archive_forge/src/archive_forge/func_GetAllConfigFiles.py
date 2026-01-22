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