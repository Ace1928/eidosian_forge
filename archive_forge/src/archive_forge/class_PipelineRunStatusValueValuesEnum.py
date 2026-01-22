from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PipelineRunStatusValueValuesEnum(_messages.Enum):
    """Pipelinerun status the user can provide. Used for cancellation.

    Values:
      PIPELINE_RUN_STATUS_UNSPECIFIED: Default enum type; should not be used.
      PIPELINE_RUN_CANCELLED: Cancelled status.
    """
    PIPELINE_RUN_STATUS_UNSPECIFIED = 0
    PIPELINE_RUN_CANCELLED = 1