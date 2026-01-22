from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsLbTrafficExtensionsGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsLbTrafficExtensionsGetRequest object.

  Fields:
    name: Required. A name of the `LbTrafficExtension` resource to get. Must
      be in the format `projects/{project}/locations/{location}/lbTrafficExten
      sions/{lb_traffic_extension}`.
  """
    name = _messages.StringField(1, required=True)