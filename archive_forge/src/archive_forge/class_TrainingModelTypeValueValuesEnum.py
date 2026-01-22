from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrainingModelTypeValueValuesEnum(_messages.Enum):
    """Optional. Type of the smart reply model. If not provided, model_type
    is used.

    Values:
      MODEL_TYPE_UNSPECIFIED: ModelType unspecified.
      SMART_REPLY_DUAL_ENCODER_MODEL: ModelType smart reply dual encoder
        model.
      SMART_REPLY_BERT_MODEL: ModelType smart reply bert model.
    """
    MODEL_TYPE_UNSPECIFIED = 0
    SMART_REPLY_DUAL_ENCODER_MODEL = 1
    SMART_REPLY_BERT_MODEL = 2