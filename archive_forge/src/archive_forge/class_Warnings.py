from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Warnings(_messages.Message):
    """Informational warning message.

  Enums:
    CodeValueValuesEnum: Output only. A warning code, if applicable.

  Messages:
    DataValue: Output only. Metadata about this warning in key: value format.
      The key should provides more detail on the warning being returned. For
      example, for warnings where there are no results in a list request for a
      particular zone, this key might be scope and the key value might be the
      zone name. Other examples might be a key indicating a deprecated
      resource and a suggested replacement.

  Fields:
    code: Output only. A warning code, if applicable.
    data: Output only. Metadata about this warning in key: value format. The
      key should provides more detail on the warning being returned. For
      example, for warnings where there are no results in a list request for a
      particular zone, this key might be scope and the key value might be the
      zone name. Other examples might be a key indicating a deprecated
      resource and a suggested replacement.
    warningMessage: Output only. A human-readable description of the warning
      code.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Output only. A warning code, if applicable.

    Values:
      WARNING_UNSPECIFIED: Default value.
      RESOURCE_NOT_ACTIVE: The policy-based route is not active and
        functioning. Common causes are the dependent network was deleted or
        the resource project was turned off.
      RESOURCE_BEING_MODIFIED: The policy-based route is being modified (e.g.
        created/deleted) at this time.
    """
        WARNING_UNSPECIFIED = 0
        RESOURCE_NOT_ACTIVE = 1
        RESOURCE_BEING_MODIFIED = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DataValue(_messages.Message):
        """Output only. Metadata about this warning in key: value format. The key
    should provides more detail on the warning being returned. For example,
    for warnings where there are no results in a list request for a particular
    zone, this key might be scope and the key value might be the zone name.
    Other examples might be a key indicating a deprecated resource and a
    suggested replacement.

    Messages:
      AdditionalProperty: An additional property for a DataValue object.

    Fields:
      additionalProperties: Additional properties of type DataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    code = _messages.EnumField('CodeValueValuesEnum', 1)
    data = _messages.MessageField('DataValue', 2)
    warningMessage = _messages.StringField(3)