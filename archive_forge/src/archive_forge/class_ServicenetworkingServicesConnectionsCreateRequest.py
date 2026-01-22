from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesConnectionsCreateRequest(_messages.Message):
    """A ServicenetworkingServicesConnectionsCreateRequest object.

  Fields:
    connection: A Connection resource to be passed as the request body.
    parent: Provider peering service that is managing peering connectivity for
      a service provider organization. For Google services that support this
      functionality it is 'services/servicenetworking.googleapis.com'.
  """
    connection = _messages.MessageField('Connection', 1)
    parent = _messages.StringField(2, required=True)