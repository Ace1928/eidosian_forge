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
def _GenerateDockerfile(self):
    """Generates a Dockerfile appropriate to this application.

    Returns:
      (ext_runtime.GeneratedFile) A file wrapper for Dockerignore
    """
    dockerfile_content = [DOCKERFILE_HEADER]
    if self.ruby_version:
        dockerfile_content.append(DOCKERFILE_CUSTOM_INTERPRETER.format(self.ruby_version))
    else:
        dockerfile_content.append(DOCKERFILE_DEFAULT_INTERPRETER)
    if self.packages:
        dockerfile_content.append(DOCKERFILE_MORE_PACKAGES.format(' '.join(self.packages)))
    else:
        dockerfile_content.append(DOCKERFILE_NO_MORE_PACKAGES)
    dockerfile_content.append(DOCKERFILE_GEM_INSTALL)
    dockerfile_content.append(DOCKERFILE_ENTRYPOINT.format(self.entrypoint))
    dockerfile = ext_runtime.GeneratedFile(config.DOCKERFILE, '\n'.join(dockerfile_content))
    return dockerfile