from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CategoryEncodingMethodValueValuesEnum(_messages.Enum):
    """Categorical feature encoding method.

    Values:
      ENCODING_METHOD_UNSPECIFIED: Unspecified encoding method.
      ONE_HOT_ENCODING: Applies one-hot encoding.
      LABEL_ENCODING: Applies label encoding.
      DUMMY_ENCODING: Applies dummy encoding.
    """
    ENCODING_METHOD_UNSPECIFIED = 0
    ONE_HOT_ENCODING = 1
    LABEL_ENCODING = 2
    DUMMY_ENCODING = 3