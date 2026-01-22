from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpConnectorsV1alphaContainerHealthDetails(_messages.Message):
    """ContainerHealthDetails reflects the health details of a container.

  Messages:
    ExtendedStatusValue: The extended status. Such as ExitCode, StartedAt,
      FinishedAt, etc.

  Fields:
    currentConfigVersion: The version of the current config.
    errorMsg: The latest error message.
    expectedConfigVersion: The version of the expected config.
    extendedStatus: The extended status. Such as ExitCode, StartedAt,
      FinishedAt, etc.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ExtendedStatusValue(_messages.Message):
        """The extended status. Such as ExitCode, StartedAt, FinishedAt, etc.

    Messages:
      AdditionalProperty: An additional property for a ExtendedStatusValue
        object.

    Fields:
      additionalProperties: Additional properties of type ExtendedStatusValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ExtendedStatusValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    currentConfigVersion = _messages.StringField(1)
    errorMsg = _messages.StringField(2)
    expectedConfigVersion = _messages.StringField(3)
    extendedStatus = _messages.MessageField('ExtendedStatusValue', 4)