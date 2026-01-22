from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomTarget(_messages.Message):
    """Information specifying a Custom Target.

  Fields:
    customTargetType: Required. The name of the CustomTargetType. Format must
      be `projects/{project}/locations/{location}/customTargetTypes/{custom_ta
      rget_type}`.
  """
    customTargetType = _messages.StringField(1)