from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterServiceperimeterV3alphaResource(_messages.Message):
    """Resource checked by service perimeter check NextTAG: 3

  Fields:
    name: The name of the resource
    permissions: The iam permission names
  """
    name = _messages.StringField(1)
    permissions = _messages.StringField(2, repeated=True)