from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.api_lib.ai.models import client as model_client
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.credentials import requests
from six.moves import http_client
def ExplainBeta(self, endpoint_ref, instances_json, args):
    """Sends online explanation request to an endpoint using v1beta1 API."""
    explain_request = self.messages.GoogleCloudAiplatformV1beta1ExplainRequest(instances=_ConvertPyListToMessageList(extra_types.JsonValue, instances_json['instances']))
    if 'parameters' in instances_json:
        explain_request.parameters = encoding.PyValueToMessage(extra_types.JsonValue, instances_json['parameters'])
    if 'explanation_spec_override' in instances_json:
        explain_request.explanationSpecOverride = encoding.PyValueToMessage(self.messages.GoogleCloudAiplatformV1beta1ExplanationSpecOverride, instances_json['explanation_spec_override'])
    if args.deployed_model_id is not None:
        explain_request.deployedModelId = args.deployed_model_id
    req = self.messages.AiplatformProjectsLocationsEndpointsExplainRequest(endpoint=endpoint_ref.RelativeName(), googleCloudAiplatformV1beta1ExplainRequest=explain_request)
    return self.client.projects_locations_endpoints.Explain(req)