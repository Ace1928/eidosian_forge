from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerResizeRequestStatus(_messages.Message):
    """A InstanceGroupManagerResizeRequestStatus object.

  Messages:
    ErrorValue: [Output only] Fatal errors encountered during the queueing or
      provisioning phases of the ResizeRequest that caused the transition to
      the FAILED state. Contrary to the last_attempt errors, this field is
      final and errors are never removed from here, as the ResizeRequest is
      not going to retry.

  Fields:
    error: [Output only] Fatal errors encountered during the queueing or
      provisioning phases of the ResizeRequest that caused the transition to
      the FAILED state. Contrary to the last_attempt errors, this field is
      final and errors are never removed from here, as the ResizeRequest is
      not going to retry.
    lastAttempt: [Output only] Information about the last attempt to fulfill
      the request. The value is temporary since the ResizeRequest can retry,
      as long as it's still active and the last attempt value can either be
      cleared or replaced with a different error. Since ResizeRequest retries
      infrequently, the value may be stale and no longer show an active
      problem. The value is cleared when ResizeRequest transitions to the
      final state (becomes inactive). If the final state is FAILED the error
      describing it will be storred in the "error" field only.
  """

    class ErrorValue(_messages.Message):
        """[Output only] Fatal errors encountered during the queueing or
    provisioning phases of the ResizeRequest that caused the transition to the
    FAILED state. Contrary to the last_attempt errors, this field is final and
    errors are never removed from here, as the ResizeRequest is not going to
    retry.

    Messages:
      ErrorsValueListEntry: A ErrorsValueListEntry object.

    Fields:
      errors: [Output Only] The array of errors encountered while processing
        this operation.
    """

        class ErrorsValueListEntry(_messages.Message):
            """A ErrorsValueListEntry object.

      Messages:
        ErrorDetailsValueListEntry: A ErrorDetailsValueListEntry object.

      Fields:
        code: [Output Only] The error type identifier for this error.
        errorDetails: [Output Only] An optional list of messages that contain
          the error details. There is a set of defined message types to use
          for providing details.The syntax depends on the error code. For
          example, QuotaExceededInfo will have details when the error code is
          QUOTA_EXCEEDED.
        location: [Output Only] Indicates the field in the request that caused
          the error. This property is optional.
        message: [Output Only] An optional, human-readable error message.
      """

            class ErrorDetailsValueListEntry(_messages.Message):
                """A ErrorDetailsValueListEntry object.

        Fields:
          errorInfo: A ErrorInfo attribute.
          help: A Help attribute.
          localizedMessage: A LocalizedMessage attribute.
          quotaInfo: A QuotaExceededInfo attribute.
        """
                errorInfo = _messages.MessageField('ErrorInfo', 1)
                help = _messages.MessageField('Help', 2)
                localizedMessage = _messages.MessageField('LocalizedMessage', 3)
                quotaInfo = _messages.MessageField('QuotaExceededInfo', 4)
            code = _messages.StringField(1)
            errorDetails = _messages.MessageField('ErrorDetailsValueListEntry', 2, repeated=True)
            location = _messages.StringField(3)
            message = _messages.StringField(4)
        errors = _messages.MessageField('ErrorsValueListEntry', 1, repeated=True)
    error = _messages.MessageField('ErrorValue', 1)
    lastAttempt = _messages.MessageField('InstanceGroupManagerResizeRequestStatusLastAttempt', 2)