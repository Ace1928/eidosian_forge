from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubnetStatus(_messages.Message):
    """SubnetStatus contains detailed and current technical information about
  this subnet resource.

  Fields:
    linkLayerAddresses: A list of LinkLayerAddress, describing the ip address
      and corresponding link-layer address of the neighbors for this subnet.
    macAddress: BVI MAC address.
    name: The name of CCFE subnet resource.
  """
    linkLayerAddresses = _messages.MessageField('LinkLayerAddress', 1, repeated=True)
    macAddress = _messages.StringField(2)
    name = _messages.StringField(3)