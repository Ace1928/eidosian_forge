from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityTypeEvalStateValueValuesEnum(_messages.Enum):
    """Details of the evaluation state of the identity type

    Values:
      IDENTITY_TYPE_EVAL_STATE_UNSPECIFIED: Not used
      IDENTITY_TYPE_EVAL_STATE_GRANTED: The request type matches the identity
      IDENTITY_TYPE_EVAL_STATE_NOT_GRANTED: The request type doesn't match the
        identity
      IDENTITY_TYPE_EVAL_STATE_NOT_SUPPORTED: The identity type is not
        supported
    """
    IDENTITY_TYPE_EVAL_STATE_UNSPECIFIED = 0
    IDENTITY_TYPE_EVAL_STATE_GRANTED = 1
    IDENTITY_TYPE_EVAL_STATE_NOT_GRANTED = 2
    IDENTITY_TYPE_EVAL_STATE_NOT_SUPPORTED = 3