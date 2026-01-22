from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsGatewaysDeleteRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsGatewaysDeleteRequest object.

  Fields:
    name: Required. A name of the Gateway to delete. Must be in the format
      `projects/*/locations/*/gateways/*`.
  """
    name = _messages.StringField(1, required=True)