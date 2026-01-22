from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SampleMethodValueValuesEnum(_messages.Enum):
    """How to sample the data.

    Values:
      SAMPLE_METHOD_UNSPECIFIED: No sampling.
      TOP: Scan from the top (default).
      RANDOM_START: For each file larger than bytes_limit_per_file, randomly
        pick the offset to start scanning. The scanned bytes are contiguous.
    """
    SAMPLE_METHOD_UNSPECIFIED = 0
    TOP = 1
    RANDOM_START = 2