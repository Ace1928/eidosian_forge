from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsEndpointPoliciesGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsEndpointPoliciesGetRequest object.

  Fields:
    name: Required. A name of the EndpointPolicy to get. Must be in the format
      `projects/*/locations/global/endpointPolicies/*`.
  """
    name = _messages.StringField(1, required=True)