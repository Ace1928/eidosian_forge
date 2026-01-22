from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1HumanReviewStatus(_messages.Message):
    """The status of human review on a processed document.

  Enums:
    StateValueValuesEnum: The state of human review on the processing request.

  Fields:
    humanReviewOperation: The name of the operation triggered by the processed
      document. This field is populated only when the state is
      `HUMAN_REVIEW_IN_PROGRESS`. It has the same response type and metadata
      as the long-running operation returned by ReviewDocument.
    state: The state of human review on the processing request.
    stateMessage: A message providing more details about the human review
      state.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of human review on the processing request.

    Values:
      STATE_UNSPECIFIED: Human review state is unspecified. Most likely due to
        an internal error.
      SKIPPED: Human review is skipped for the document. This can happen
        because human review isn't enabled on the processor or the processing
        request has been set to skip this document.
      VALIDATION_PASSED: Human review validation is triggered and passed, so
        no review is needed.
      IN_PROGRESS: Human review validation is triggered and the document is
        under review.
      ERROR: Some error happened during triggering human review, see the
        state_message for details.
    """
        STATE_UNSPECIFIED = 0
        SKIPPED = 1
        VALIDATION_PASSED = 2
        IN_PROGRESS = 3
        ERROR = 4
    humanReviewOperation = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)
    stateMessage = _messages.StringField(3)