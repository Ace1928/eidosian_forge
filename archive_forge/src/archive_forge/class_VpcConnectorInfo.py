from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpcConnectorInfo(_messages.Message):
    """For display only. Metadata associated with a VPC connector.

  Fields:
    displayName: Name of a VPC connector.
    location: Location in which the VPC connector is deployed.
    uri: URI of a VPC connector.
  """
    displayName = _messages.StringField(1)
    location = _messages.StringField(2)
    uri = _messages.StringField(3)