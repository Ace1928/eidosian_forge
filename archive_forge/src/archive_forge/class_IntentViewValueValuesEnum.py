from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IntentViewValueValuesEnum(_messages.Enum):
    """Optional. The resource view to apply to the returned intent.

    Values:
      INTENT_VIEW_UNSPECIFIED: Training phrases field is not populated in the
        response.
      INTENT_VIEW_FULL: All fields are populated.
    """
    INTENT_VIEW_UNSPECIFIED = 0
    INTENT_VIEW_FULL = 1