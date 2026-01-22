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
def _GenerateAppYaml(self):
    """Generates an app.yaml file appropriate to this application.

    Returns:
      (ext_runtime.GeneratedFile) A file wrapper for app.yaml
    """
    app_yaml = os.path.join(self.root, 'app.yaml')
    runtime = 'custom' if self.params.custom else 'ruby'
    app_yaml_contents = APP_YAML_CONTENTS.format(runtime=runtime, entrypoint=self.entrypoint)
    app_yaml = ext_runtime.GeneratedFile('app.yaml', app_yaml_contents)
    return app_yaml