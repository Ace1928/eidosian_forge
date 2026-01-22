from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalysisStatusValueValuesEnum(_messages.Enum):
    """The status of discovery for the resource.

    Values:
      ANALYSIS_STATUS_UNSPECIFIED: Unknown.
      PENDING: Resource is known but no action has been taken yet.
      SCANNING: Resource is being analyzed.
      FINISHED_SUCCESS: Analysis has finished successfully.
      COMPLETE: Analysis has completed.
      FINISHED_FAILED: Analysis has finished unsuccessfully, the analysis
        itself is in a bad state.
      FINISHED_UNSUPPORTED: The resource is known not to be supported.
    """
    ANALYSIS_STATUS_UNSPECIFIED = 0
    PENDING = 1
    SCANNING = 2
    FINISHED_SUCCESS = 3
    COMPLETE = 4
    FINISHED_FAILED = 5
    FINISHED_UNSUPPORTED = 6