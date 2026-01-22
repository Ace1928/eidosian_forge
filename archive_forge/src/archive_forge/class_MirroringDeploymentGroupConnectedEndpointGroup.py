from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MirroringDeploymentGroupConnectedEndpointGroup(_messages.Message):
    """An endpoint group connected to this deployment group.

  Fields:
    name: Output only. A connected mirroring endpoint group.
  """
    name = _messages.StringField(1)