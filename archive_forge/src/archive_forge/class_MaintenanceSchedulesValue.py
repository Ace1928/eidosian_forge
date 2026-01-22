from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MaintenanceSchedulesValue(_messages.Message):
    """The MaintenanceSchedule contains the scheduling information of
    published maintenance schedule with same key as software_versions.

    Messages:
      AdditionalProperty: An additional property for a
        MaintenanceSchedulesValue object.

    Fields:
      additionalProperties: Additional properties of type
        MaintenanceSchedulesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MaintenanceSchedulesValue object.

      Fields:
        key: Name of the additional property.
        value: A
          GoogleCloudSaasacceleratorManagementProvidersV1MaintenanceSchedule
          attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudSaasacceleratorManagementProvidersV1MaintenanceSchedule', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)