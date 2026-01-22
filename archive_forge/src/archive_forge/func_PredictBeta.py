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
def PredictBeta(self, endpoint_ref, instances_json):
    """Sends online prediction request to an endpoint using v1beta1 API."""
    predict_request = self.messages.GoogleCloudAiplatformV1beta1PredictRequest(instances=_ConvertPyListToMessageList(extra_types.JsonValue, instances_json['instances']))
    if 'parameters' in instances_json:
        predict_request.parameters = encoding.PyValueToMessage(extra_types.JsonValue, instances_json['parameters'])
    req = self.messages.AiplatformProjectsLocationsEndpointsPredictRequest(endpoint=endpoint_ref.RelativeName(), googleCloudAiplatformV1beta1PredictRequest=predict_request)
    return self.client.projects_locations_endpoints.Predict(req)