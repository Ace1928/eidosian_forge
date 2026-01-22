from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClassificationModelOptions(_messages.Message):
    """Model options available for classification requests.

  Fields:
    v1Model: Setting this field will use the V1 model and V1 content
      categories version. The V1 model is a legacy model; support for this
      will be discontinued in the future.
    v2Model: Setting this field will use the V2 model with the appropriate
      content categories version. The V2 model is a better performing model.
  """
    v1Model = _messages.MessageField('V1Model', 1)
    v2Model = _messages.MessageField('V2Model', 2)