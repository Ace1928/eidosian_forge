from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedInstanceLastAttempt(_messages.Message):
    """A ManagedInstanceLastAttempt object.

  Messages:
    ErrorsValue: [Output Only] Encountered errors during the last attempt to
      create or delete the instance.

  Fields:
    errors: [Output Only] Encountered errors during the last attempt to create
      or delete the instance.
  """

    class ErrorsValue(_messages.Message):
        """[Output Only] Encountered errors during the last attempt to create or
    delete the instance.

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
    errors = _messages.MessageField('ErrorsValue', 1)