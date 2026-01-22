from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeNetworkEndpointGroupsGetRequest(_messages.Message):
    """A ComputeNetworkEndpointGroupsGetRequest object.

  Fields:
    networkEndpointGroup: The name of the network endpoint group. It should
      comply with RFC1035.
    project: Project ID for this request.
    zone: The name of the zone where the network endpoint group is located. It
      should comply with RFC1035.
  """
    networkEndpointGroup = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    zone = _messages.StringField(3, required=True)