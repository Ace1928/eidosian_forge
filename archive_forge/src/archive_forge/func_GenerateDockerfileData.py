from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def GenerateDockerfileData(self):
    """Generates dockerfiles.

    Returns:
      list(ext_runtime.GeneratedFile) the list of generated dockerfiles
    """
    if self.runtime == 'python-compat':
        dockerfile_preamble = COMPAT_DOCKERFILE_PREAMBLE
    else:
        dockerfile_preamble = PYTHON27_DOCKERFILE_PREAMBLE
    all_config_files = []
    dockerfile_name = config.DOCKERFILE
    dockerfile_components = [dockerfile_preamble, DOCKERFILE_INSTALL_APP]
    if self.runtime == 'python-compat':
        dockerfile_components.append(DOCKERFILE_INSTALL_REQUIREMENTS_TXT)
    dockerfile_contents = ''.join((c for c in dockerfile_components))
    dockerfile = ext_runtime.GeneratedFile(dockerfile_name, dockerfile_contents)
    all_config_files.append(dockerfile)
    dockerignore = ext_runtime.GeneratedFile('.dockerignore', DOCKERIGNORE)
    all_config_files.append(dockerignore)
    return all_config_files