from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrimaryInstanceSettings(_messages.Message):
    """Settings for the cluster's primary instance

  Messages:
    DatabaseFlagsValue: Database flags to pass to AlloyDB when DMS is creating
      the AlloyDB cluster and instances. See the AlloyDB documentation for how
      these can be used.
    LabelsValue: Labels for the AlloyDB primary instance created by DMS. An
      object containing a list of 'key', 'value' pairs.

  Fields:
    databaseFlags: Database flags to pass to AlloyDB when DMS is creating the
      AlloyDB cluster and instances. See the AlloyDB documentation for how
      these can be used.
    id: Required. The ID of the AlloyDB primary instance. The ID must satisfy
      the regex expression "[a-z0-9-]+".
    labels: Labels for the AlloyDB primary instance created by DMS. An object
      containing a list of 'key', 'value' pairs.
    machineConfig: Configuration for the machines that host the underlying
      database engine.
    privateIp: Output only. The private IP address for the Instance. This is
      the connection endpoint for an end-user application.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DatabaseFlagsValue(_messages.Message):
        """Database flags to pass to AlloyDB when DMS is creating the AlloyDB
    cluster and instances. See the AlloyDB documentation for how these can be
    used.

    Messages:
      AdditionalProperty: An additional property for a DatabaseFlagsValue
        object.

    Fields:
      additionalProperties: Additional properties of type DatabaseFlagsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DatabaseFlagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels for the AlloyDB primary instance created by DMS. An object
    containing a list of 'key', 'value' pairs.

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
    databaseFlags = _messages.MessageField('DatabaseFlagsValue', 1)
    id = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    machineConfig = _messages.MessageField('MachineConfig', 4)
    privateIp = _messages.StringField(5)