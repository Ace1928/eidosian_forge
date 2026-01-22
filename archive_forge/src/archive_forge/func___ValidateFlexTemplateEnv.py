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
def __ValidateFlexTemplateEnv(env, sdk_language):
    """Builds and validates Flex template environment values.

    Args:
      env: Dictionary of env variables to set in the container image.
      sdk_language: SDK language of the flex template.

    Returns:
      True on valid env values.

    Raises:
      ValueError: If is any of parameter value is invalid.
    """
    if sdk_language == 'JAVA' and 'FLEX_TEMPLATE_JAVA_MAIN_CLASS' not in env:
        raise ValueError('FLEX_TEMPLATE_JAVA_MAIN_CLASS environment variable should be provided for all JAVA jobs.')
    elif sdk_language == 'PYTHON' and 'FLEX_TEMPLATE_PYTHON_PY_FILE' not in env:
        raise ValueError('FLEX_TEMPLATE_PYTHON_PY_FILE environment variable should be provided for all PYTHON jobs.')
    elif sdk_language == 'GO' and 'FLEX_TEMPLATE_GO_BINARY' not in env:
        raise ValueError('FLEX_TEMPLATE_GO_BINARY environment variable should be provided for all GO jobs.')
    return True