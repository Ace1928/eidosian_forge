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
def _ValidateTemplateParameters(parameters):
    """Validates ParameterMetadata objects in template metadata.

    Args:
      parameters: List of ParameterMetadata objects.

    Raises:
      ValueError: If is any of the required field is not set.
    """
    for parameter in parameters:
        if not parameter.name:
            raise ValueError('Invalid template metadata. Parameter name field is empty. Parameter: {}'.format(parameter))
        if not parameter.label:
            raise ValueError('Invalid template metadata. Parameter label field is empty. Parameter: {}'.format(parameter))
        if not parameter.helpText:
            raise ValueError('Invalid template metadata. Parameter helpText field is empty. Parameter: {}'.format(parameter))