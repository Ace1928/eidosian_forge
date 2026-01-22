from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirebasedataconnectProjectsLocationsServicesExecuteGraphqlRequest(_messages.Message):
    """A FirebasedataconnectProjectsLocationsServicesExecuteGraphqlRequest
  object.

  Fields:
    graphqlRequest: A GraphqlRequest resource to be passed as the request
      body.
    name: Required. The relative resource name of Firebase Data Connect
      service, in the format: ```
      projects/{project}/locations/{location}/services/{service} ```
  """
    graphqlRequest = _messages.MessageField('GraphqlRequest', 1)
    name = _messages.StringField(2, required=True)