from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReportInstanceInfoRequest(_messages.Message):
    """Request for notebook instances to report information to Notebooks API.

  Messages:
    MetadataValue: The metadata reported to Notebooks API. This will be merged
      to the instance metadata store

  Fields:
    metadata: The metadata reported to Notebooks API. This will be merged to
      the instance metadata store
    vmId: Required. The VM hardware token for authenticating the VM.
      https://cloud.google.com/compute/docs/instances/verifying-instance-
      identity
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """The metadata reported to Notebooks API. This will be merged to the
    instance metadata store

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Additional properties of type MetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    metadata = _messages.MessageField('MetadataValue', 1)
    vmId = _messages.StringField(2)