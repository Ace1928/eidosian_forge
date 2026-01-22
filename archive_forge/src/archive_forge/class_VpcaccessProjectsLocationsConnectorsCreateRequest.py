from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpcaccessProjectsLocationsConnectorsCreateRequest(_messages.Message):
    """A VpcaccessProjectsLocationsConnectorsCreateRequest object.

  Fields:
    connector: A Connector resource to be passed as the request body.
    connectorId: Required. The ID to use for this connector.
    parent: Required. The project ID and location in which the configuration
      should be created, specified in the format `projects/*/locations/*`.
  """
    connector = _messages.MessageField('Connector', 1)
    connectorId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)