from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicedirectoryProjectsLocationsNamespacesServicesEndpointsPatchRequest(_messages.Message):
    """A
  ServicedirectoryProjectsLocationsNamespacesServicesEndpointsPatchRequest
  object.

  Fields:
    endpoint: A Endpoint resource to be passed as the request body.
    name: Immutable. The resource name for the endpoint in the format
      `projects/*/locations/*/namespaces/*/services/*/endpoints/*`.
    updateMask: Required. List of fields to be updated in this request.
  """
    endpoint = _messages.MessageField('Endpoint', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)