from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdsProjectsLocationsEndpointsGetRequest(_messages.Message):
    """A IdsProjectsLocationsEndpointsGetRequest object.

  Fields:
    name: Required. The name of the endpoint to retrieve. Format:
      projects/{project}/locations/{location}/endpoints/{endpoint}
  """
    name = _messages.StringField(1, required=True)