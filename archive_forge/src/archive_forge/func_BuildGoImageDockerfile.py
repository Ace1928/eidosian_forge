from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import shutil
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.command_lib.builds import submit_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
@staticmethod
def BuildGoImageDockerfile(flex_template_base_image, pipeline_paths, env):
    """Builds Dockerfile contents for go flex template image.

    Args:
      flex_template_base_image: SDK version or base image to use.
      pipeline_paths: Path to pipeline binary.
      env: Dictionary of env variables to set in the container image.

    Returns:
      Dockerfile contents as string.
    """
    dockerfile_template = '\n    FROM {base_image}\n\n    {env}\n\n    {copy}\n    '
    env['FLEX_TEMPLATE_GO_BINARY'] = '/template/{}'.format(env['FLEX_TEMPLATE_GO_BINARY'])
    paths = ' '.join(pipeline_paths)
    copy_command = 'COPY {} /template/'.format(paths)
    envs = ['ENV {}={}'.format(var, val) for var, val in sorted(env.items())]
    env_list = '\n'.join(envs)
    dockerfile_contents = textwrap.dedent(dockerfile_template).format(base_image=Templates._GetFlexTemplateBaseImage(flex_template_base_image), env=env_list, copy=copy_command)
    return dockerfile_contents