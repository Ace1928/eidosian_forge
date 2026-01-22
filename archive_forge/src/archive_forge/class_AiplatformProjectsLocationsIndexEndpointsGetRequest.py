from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsIndexEndpointsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsIndexEndpointsGetRequest object.

  Fields:
    name: Required. The name of the IndexEndpoint resource. Format:
      `projects/{project}/locations/{location}/indexEndpoints/{index_endpoint}
      `
  """
    name = _messages.StringField(1, required=True)