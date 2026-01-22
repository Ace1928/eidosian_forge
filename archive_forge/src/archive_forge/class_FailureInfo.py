from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FailureInfo(_messages.Message):
    """A fatal problem encountered during the execution of the build.

  Enums:
    TypeValueValuesEnum: The name of the failure.

  Fields:
    detail: Explains the failure issue in more detail using hard-coded text.
    type: The name of the failure.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The name of the failure.

    Values:
      FAILURE_TYPE_UNSPECIFIED: Type unspecified
      PUSH_FAILED: Unable to push the image to the repository.
      PUSH_IMAGE_NOT_FOUND: Final image not found.
      PUSH_NOT_AUTHORIZED: Unauthorized push of the final image.
      LOGGING_FAILURE: Backend logging failures. Should retry.
      USER_BUILD_STEP: A build step has failed.
      FETCH_SOURCE_FAILED: The source fetching has failed.
    """
        FAILURE_TYPE_UNSPECIFIED = 0
        PUSH_FAILED = 1
        PUSH_IMAGE_NOT_FOUND = 2
        PUSH_NOT_AUTHORIZED = 3
        LOGGING_FAILURE = 4
        USER_BUILD_STEP = 5
        FETCH_SOURCE_FAILED = 6
    detail = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)