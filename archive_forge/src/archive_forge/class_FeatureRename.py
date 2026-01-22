from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FeatureRename(_messages.Message):
    """JSON request template for renaming a feature.

  Fields:
    newName: New name of the feature.
  """
    newName = _messages.StringField(1)