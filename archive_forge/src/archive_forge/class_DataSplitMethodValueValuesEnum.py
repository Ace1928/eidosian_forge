from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataSplitMethodValueValuesEnum(_messages.Enum):
    """The data split type for training and evaluation, e.g. RANDOM.

    Values:
      DATA_SPLIT_METHOD_UNSPECIFIED: Default value.
      RANDOM: Splits data randomly.
      CUSTOM: Splits data with the user provided tags.
      SEQUENTIAL: Splits data sequentially.
      NO_SPLIT: Data split will be skipped.
      AUTO_SPLIT: Splits data automatically: Uses NO_SPLIT if the data size is
        small. Otherwise uses RANDOM.
    """
    DATA_SPLIT_METHOD_UNSPECIFIED = 0
    RANDOM = 1
    CUSTOM = 2
    SEQUENTIAL = 3
    NO_SPLIT = 4
    AUTO_SPLIT = 5