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
def BuildPythonImageDockerfile(flex_template_base_image, pipeline_paths, env):
    """Builds Dockerfile contents for python flex template image.

    Args:
      flex_template_base_image: SDK version or base image to use.
      pipeline_paths: List of paths to pipelines and dependencies.
      env: Dictionary of env variables to set in the container image.

    Returns:
      Dockerfile contents as string.
    """
    dockerfile_template = '\n    FROM {base_image}\n\n    {env}\n\n    {copy}\n\n    {commands}\n    '
    commands = ['apt-get update', 'apt-get install -y libffi-dev git', 'rm -rf /var/lib/apt/lists/*']
    env['FLEX_TEMPLATE_PYTHON_PY_FILE'] = '/template/{}'.format(env['FLEX_TEMPLATE_PYTHON_PY_FILE'])
    if 'FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE' in env:
        env['FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE'] = '/template/{}'.format(env['FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE'])
        commands.append('pip install --no-cache-dir -U -r {}'.format(env['FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE']))
    if 'FLEX_TEMPLATE_PYTHON_SETUP_FILE' in env:
        env['FLEX_TEMPLATE_PYTHON_SETUP_FILE'] = '/template/{}'.format(env['FLEX_TEMPLATE_PYTHON_SETUP_FILE'])
    envs = ['ENV {}={}'.format(k, v) for k, v in sorted(env.items())]
    env_list = '\n'.join(envs)
    paths = ' '.join(pipeline_paths)
    copy_command = 'COPY {} /template/'.format(paths)
    dockerfile_contents = textwrap.dedent(dockerfile_template).format(base_image=Templates._GetFlexTemplateBaseImage(flex_template_base_image), env=env_list, copy=copy_command, commands='RUN ' + ' && '.join(commands))
    return dockerfile_contents