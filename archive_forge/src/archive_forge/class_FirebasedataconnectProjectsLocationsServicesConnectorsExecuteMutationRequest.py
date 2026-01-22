from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirebasedataconnectProjectsLocationsServicesConnectorsExecuteMutationRequest(_messages.Message):
    """A
  FirebasedataconnectProjectsLocationsServicesConnectorsExecuteMutationRequest
  object.

  Fields:
    executeMutationRequest: A ExecuteMutationRequest resource to be passed as
      the request body.
    name: Required. The resource name of the connector to find the predefined
      mutation, in the format: ``` projects/{project}/locations/{location}/ser
      vices/{service}/connectors/{connector} ```
  """
    executeMutationRequest = _messages.MessageField('ExecuteMutationRequest', 1)
    name = _messages.StringField(2, required=True)