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
def BuildDockerfile(flex_template_base_image, pipeline_paths, env, sdk_language):
    """Builds Dockerfile contents for flex template image.

    Args:
      flex_template_base_image: SDK version or base image to use.
      pipeline_paths: List of paths to pipelines and dependencies.
      env: Dictionary of env variables to set in the container image.
      sdk_language: SDK language of the flex template.

    Returns:
      Dockerfile contents as string.
    """
    if sdk_language == 'JAVA':
        return Templates.BuildJavaImageDockerfile(flex_template_base_image, pipeline_paths, env)
    elif sdk_language == 'PYTHON':
        return Templates.BuildPythonImageDockerfile(flex_template_base_image, pipeline_paths, env)
    elif sdk_language == 'GO':
        return Templates.BuildGoImageDockerfile(flex_template_base_image, pipeline_paths, env)