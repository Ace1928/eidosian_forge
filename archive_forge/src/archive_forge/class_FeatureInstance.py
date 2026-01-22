from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FeatureInstance(_messages.Message):
    """JSON template for a "feature instance".

  Fields:
    feature: The feature that this is an instance of. A calendar resource may
      have multiple instances of a feature.
  """
    feature = _messages.MessageField('Feature', 1)