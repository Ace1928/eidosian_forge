from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzedWorkloadTypeValueValuesEnum(_messages.Enum):
    """Output only. Type of the workload being analyzed.

    Values:
      WORKLOAD_TYPE_UNSPECIFIED: Undefined option
      BATCH: Serverless batch job
    """
    WORKLOAD_TYPE_UNSPECIFIED = 0
    BATCH = 1