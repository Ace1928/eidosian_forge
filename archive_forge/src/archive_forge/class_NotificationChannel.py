from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotificationChannel(_messages.Message):
    """A NotificationChannel is a medium through which an alert is delivered
  when a policy violation is detected. Examples of channels include email,
  SMS, and third-party messaging applications. Fields containing sensitive
  information like authentication tokens or contact info are only partially
  populated on retrieval.

  Enums:
    VerificationStatusValueValuesEnum: Indicates whether this channel has been
      verified or not. On a ListNotificationChannels or GetNotificationChannel
      operation, this field is expected to be populated.If the value is
      UNVERIFIED, then it indicates that the channel is non-functioning (it
      both requires verification and lacks verification); otherwise, it is
      assumed that the channel works.If the channel is neither VERIFIED nor
      UNVERIFIED, it implies that the channel is of a type that does not
      require verification or that this specific channel has been exempted
      from verification because it was created prior to verification being
      required for channels of this type.This field cannot be modified using a
      standard UpdateNotificationChannel operation. To change the value of
      this field, you must call VerifyNotificationChannel.

  Messages:
    LabelsValue: Configuration fields that define the channel and its
      behavior. The permissible and required labels are specified in the
      NotificationChannelDescriptor.labels of the
      NotificationChannelDescriptor corresponding to the type field.
    UserLabelsValue: User-supplied key/value data that does not need to
      conform to the corresponding NotificationChannelDescriptor's schema,
      unlike the labels field. This field is intended to be used for
      organizing and identifying the NotificationChannel objects.The field can
      contain up to 64 entries. Each key and value is limited to 63 Unicode
      characters or 128 bytes, whichever is smaller. Labels and values can
      contain only lowercase letters, numerals, underscores, and dashes. Keys
      must begin with a letter.

  Fields:
    creationRecord: Record of the creation of this channel.
    description: An optional human-readable description of this notification
      channel. This description may provide additional details, beyond the
      display name, for the channel. This may not exceed 1024 Unicode
      characters.
    displayName: An optional human-readable name for this notification
      channel. It is recommended that you specify a non-empty and unique name
      in order to make it easier to identify the channels in your project,
      though this is not enforced. The display name is limited to 512 Unicode
      characters.
    enabled: Whether notifications are forwarded to the described channel.
      This makes it possible to disable delivery of notifications to a
      particular channel without removing the channel from all alerting
      policies that reference the channel. This is a more convenient approach
      when the change is temporary and you want to receive notifications from
      the same set of alerting policies on the channel at some point in the
      future.
    labels: Configuration fields that define the channel and its behavior. The
      permissible and required labels are specified in the
      NotificationChannelDescriptor.labels of the
      NotificationChannelDescriptor corresponding to the type field.
    mutationRecords: Records of the modification of this channel.
    name: The full REST resource name for this channel. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/notificationChannels/[CHANNEL_ID] The
      [CHANNEL_ID] is automatically assigned by the server on creation.
    type: The type of the notification channel. This field matches the value
      of the NotificationChannelDescriptor.type field.
    userLabels: User-supplied key/value data that does not need to conform to
      the corresponding NotificationChannelDescriptor's schema, unlike the
      labels field. This field is intended to be used for organizing and
      identifying the NotificationChannel objects.The field can contain up to
      64 entries. Each key and value is limited to 63 Unicode characters or
      128 bytes, whichever is smaller. Labels and values can contain only
      lowercase letters, numerals, underscores, and dashes. Keys must begin
      with a letter.
    verificationStatus: Indicates whether this channel has been verified or
      not. On a ListNotificationChannels or GetNotificationChannel operation,
      this field is expected to be populated.If the value is UNVERIFIED, then
      it indicates that the channel is non-functioning (it both requires
      verification and lacks verification); otherwise, it is assumed that the
      channel works.If the channel is neither VERIFIED nor UNVERIFIED, it
      implies that the channel is of a type that does not require verification
      or that this specific channel has been exempted from verification
      because it was created prior to verification being required for channels
      of this type.This field cannot be modified using a standard
      UpdateNotificationChannel operation. To change the value of this field,
      you must call VerifyNotificationChannel.
  """

    class VerificationStatusValueValuesEnum(_messages.Enum):
        """Indicates whether this channel has been verified or not. On a
    ListNotificationChannels or GetNotificationChannel operation, this field
    is expected to be populated.If the value is UNVERIFIED, then it indicates
    that the channel is non-functioning (it both requires verification and
    lacks verification); otherwise, it is assumed that the channel works.If
    the channel is neither VERIFIED nor UNVERIFIED, it implies that the
    channel is of a type that does not require verification or that this
    specific channel has been exempted from verification because it was
    created prior to verification being required for channels of this
    type.This field cannot be modified using a standard
    UpdateNotificationChannel operation. To change the value of this field,
    you must call VerifyNotificationChannel.

    Values:
      VERIFICATION_STATUS_UNSPECIFIED: Sentinel value used to indicate that
        the state is unknown, omitted, or is not applicable (as in the case of
        channels that neither support nor require verification in order to
        function).
      UNVERIFIED: The channel has yet to be verified and requires verification
        to function. Note that this state also applies to the case where the
        verification process has been initiated by sending a verification code
        but where the verification code has not been submitted to complete the
        process.
      VERIFIED: It has been proven that notifications can be received on this
        notification channel and that someone on the project has access to
        messages that are delivered to that channel.
    """
        VERIFICATION_STATUS_UNSPECIFIED = 0
        UNVERIFIED = 1
        VERIFIED = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Configuration fields that define the channel and its behavior. The
    permissible and required labels are specified in the
    NotificationChannelDescriptor.labels of the NotificationChannelDescriptor
    corresponding to the type field.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UserLabelsValue(_messages.Message):
        """User-supplied key/value data that does not need to conform to the
    corresponding NotificationChannelDescriptor's schema, unlike the labels
    field. This field is intended to be used for organizing and identifying
    the NotificationChannel objects.The field can contain up to 64 entries.
    Each key and value is limited to 63 Unicode characters or 128 bytes,
    whichever is smaller. Labels and values can contain only lowercase
    letters, numerals, underscores, and dashes. Keys must begin with a letter.

    Messages:
      AdditionalProperty: An additional property for a UserLabelsValue object.

    Fields:
      additionalProperties: Additional properties of type UserLabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a UserLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    creationRecord = _messages.MessageField('MutationRecord', 1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    enabled = _messages.BooleanField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    mutationRecords = _messages.MessageField('MutationRecord', 6, repeated=True)
    name = _messages.StringField(7)
    type = _messages.StringField(8)
    userLabels = _messages.MessageField('UserLabelsValue', 9)
    verificationStatus = _messages.EnumField('VerificationStatusValueValuesEnum', 10)