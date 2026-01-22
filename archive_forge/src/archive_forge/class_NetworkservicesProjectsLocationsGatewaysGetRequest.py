from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsGatewaysGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsGatewaysGetRequest object.

  Fields:
    name: Required. A name of the Gateway to get. Must be in the format
      `projects/*/locations/*/gateways/*`.
  """
    name = _messages.StringField(1, required=True)