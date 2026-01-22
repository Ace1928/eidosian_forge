from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReleaseRenderStateValueValuesEnum(_messages.Enum):
    """The state of the release render.

    Values:
      RENDER_STATE_UNSPECIFIED: The render state is unspecified.
      SUCCEEDED: All rendering operations have completed successfully.
      FAILED: All rendering operations have completed, and one or more have
        failed.
      IN_PROGRESS: Rendering has started and is not complete.
    """
    RENDER_STATE_UNSPECIFIED = 0
    SUCCEEDED = 1
    FAILED = 2
    IN_PROGRESS = 3