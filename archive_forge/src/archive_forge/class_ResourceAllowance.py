from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceAllowance(_messages.Message):
    """The Resource Allowance description for Cloud Batch. Only one Resource
  Allowance is supported now under a specific location and project.

  Messages:
    LabelsValue: Optional. Labels are attributes that can be set and used by
      both the user and by Batch. Labels must meet the following constraints:
      * Keys and values can contain only lowercase letters, numeric
      characters, underscores, and dashes. * All characters must use UTF-8
      encoding, and international characters are allowed. * Keys must start
      with a lowercase letter or international character. * Each resource is
      limited to a maximum of 64 labels. Both keys and values are additionally
      constrained to be <= 128 bytes.

  Fields:
    createTime: Output only. Time when the ResourceAllowance was created.
    labels: Optional. Labels are attributes that can be set and used by both
      the user and by Batch. Labels must meet the following constraints: *
      Keys and values can contain only lowercase letters, numeric characters,
      underscores, and dashes. * All characters must use UTF-8 encoding, and
      international characters are allowed. * Keys must start with a lowercase
      letter or international character. * Each resource is limited to a
      maximum of 64 labels. Both keys and values are additionally constrained
      to be <= 128 bytes.
    name: Identifier. ResourceAllowance name. For example:
      "projects/123456/locations/us-central1/resourceAllowances/resource-
      allowance-1".
    notifications: Optional. Notification configurations.
    uid: Output only. A system generated unique ID (in UUID4 format) for the
      ResourceAllowance.
    usageResourceAllowance: The detail of usage resource allowance.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels are attributes that can be set and used by both the
    user and by Batch. Labels must meet the following constraints: * Keys and
    values can contain only lowercase letters, numeric characters,
    underscores, and dashes. * All characters must use UTF-8 encoding, and
    international characters are allowed. * Keys must start with a lowercase
    letter or international character. * Each resource is limited to a maximum
    of 64 labels. Both keys and values are additionally constrained to be <=
    128 bytes.

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
    createTime = _messages.StringField(1)
    labels = _messages.MessageField('LabelsValue', 2)
    name = _messages.StringField(3)
    notifications = _messages.MessageField('Notification', 4, repeated=True)
    uid = _messages.StringField(5)
    usageResourceAllowance = _messages.MessageField('UsageResourceAllowance', 6)