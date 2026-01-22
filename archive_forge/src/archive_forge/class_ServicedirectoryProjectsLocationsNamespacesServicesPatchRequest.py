from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicedirectoryProjectsLocationsNamespacesServicesPatchRequest(_messages.Message):
    """A ServicedirectoryProjectsLocationsNamespacesServicesPatchRequest
  object.

  Fields:
    name: Immutable. The resource name for the service in the format
      `projects/*/locations/*/namespaces/*/services/*`.
    service: A Service resource to be passed as the request body.
    updateMask: Required. List of fields to be updated in this request.
  """
    name = _messages.StringField(1, required=True)
    service = _messages.MessageField('Service', 2)
    updateMask = _messages.StringField(3)