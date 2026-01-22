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
def _BuildTemplateMetadata(template_metadata_json):
    """Builds and validates TemplateMetadata object.

    Args:
      template_metadata_json: Template metadata in json format.

    Returns:
      TemplateMetadata object on success.

    Raises:
      ValueError: If is any of the required field is not set.
    """
    template_metadata = encoding.JsonToMessage(Templates.TEMPLATE_METADATA, template_metadata_json)
    template_metadata_obj = Templates.TEMPLATE_METADATA()
    if not template_metadata.name:
        raise ValueError('Invalid template metadata. Name field is empty. Template Metadata: {}'.format(template_metadata))
    template_metadata_obj.name = template_metadata.name
    if template_metadata.description:
        template_metadata_obj.description = template_metadata.description
    if template_metadata.parameters:
        Templates._ValidateTemplateParameters(template_metadata.parameters)
        template_metadata_obj.parameters = template_metadata.parameters
    return template_metadata_obj