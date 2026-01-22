from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReadConsistencyValueValuesEnum(_messages.Enum):
    """The non-transactional read consistency to use.

    Values:
      READ_CONSISTENCY_UNSPECIFIED: Unspecified. This value must not be used.
      STRONG: Strong consistency.
      EVENTUAL: Eventual consistency.
    """
    READ_CONSISTENCY_UNSPECIFIED = 0
    STRONG = 1
    EVENTUAL = 2