from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VariantValueValuesEnum(_messages.Enum):
    """Logging variant deployed on nodes.

    Values:
      VARIANT_UNSPECIFIED: Default value. This shouldn't be used.
      DEFAULT: default logging variant.
      MAX_THROUGHPUT: maximum logging throughput variant.
    """
    VARIANT_UNSPECIFIED = 0
    DEFAULT = 1
    MAX_THROUGHPUT = 2