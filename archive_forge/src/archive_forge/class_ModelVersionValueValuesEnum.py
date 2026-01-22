from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModelVersionValueValuesEnum(_messages.Enum):
    """The version of the model used to create these annotations.

    Values:
      VERSION_UNSPECIFIED: <no description>
      INDEXING_20181017: <no description>
      INDEXING_20191206: <no description>
      INDEXING_20200313: <no description>
      INDEXING_20210618: <no description>
      STANDARD_20220516: <no description>
    """
    VERSION_UNSPECIFIED = 0
    INDEXING_20181017 = 1
    INDEXING_20191206 = 2
    INDEXING_20200313 = 3
    INDEXING_20210618 = 4
    STANDARD_20220516 = 5