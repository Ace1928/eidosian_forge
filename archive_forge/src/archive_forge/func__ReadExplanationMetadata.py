from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.ai import operations
from googlecloudsdk.api_lib.ai.models import client
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import models_util
from googlecloudsdk.command_lib.ai import operations_util
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.core import yaml
def _ReadExplanationMetadata(self, explanation_metadata_file):
    explanation_metadata = None
    if not explanation_metadata_file:
        return explanation_metadata
    data = yaml.load_path(explanation_metadata_file)
    if data:
        explanation_metadata = messages_util.DictToMessageWithErrorCheck(data, self.messages.GoogleCloudAiplatformV1beta1ExplanationMetadata)
    return explanation_metadata