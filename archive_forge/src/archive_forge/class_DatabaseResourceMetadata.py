from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatabaseResourceMetadata(_messages.Message):
    """Common model for database resource instance metadata.

  Enums:
    CurrentStateValueValuesEnum: Current state of the instance.
    ExpectedStateValueValuesEnum: The state that the instance is expected to
      be in. For example, an instance state can transition to UNHEALTHY due to
      wrong patch update, while the expected state will remain at the HEALTHY.
    InstanceTypeValueValuesEnum: The type of the instance. Specified at
      creation time.

  Messages:
    UserLabelsValue: User-provided labels, represented as a dictionary where
      each label is a single key value pair.

  Fields:
    availabilityConfiguration: Availability configuration for this instance
    backupConfiguration: Backup configuration for this instance
    backupRun: Latest backup run information for this instance
    creationTime: The creation time of the resource, i.e. the time when
      resource is created and recorded in partner service.
    currentState: Current state of the instance.
    customMetadata: Any custom metadata associated with the resource
    entitlements: Entitlements associated with the resource
    expectedState: The state that the instance is expected to be in. For
      example, an instance state can transition to UNHEALTHY due to wrong
      patch update, while the expected state will remain at the HEALTHY.
    id: Required. Unique identifier for a Database resource
    instanceType: The type of the instance. Specified at creation time.
    location: The resource location. REQUIRED
    primaryResourceId: Identifier for this resource's immediate parent/primary
      resource if the current resource is a replica or derived form of another
      Database resource. Else it would be NULL. REQUIRED if the immediate
      parent exists when first time resource is getting ingested, otherwise
      optional.
    product: The product this resource represents.
    resourceContainer: Closest parent Cloud Resource Manager container of this
      resource. It must be resource name of a Cloud Resource Manager project
      with the format of "/", such as "projects/123". For GCP provided
      resources, number should be project number.
    resourceName: Required. Different from DatabaseResourceId.unique_id, a
      resource name can be reused over time. That is, after a resource named
      "ABC" is deleted, the name "ABC" can be used to to create a new resource
      within the same source. Resource name to follow CAIS resource_name
      format as noted here go/condor-common-datamodel
    updationTime: The time at which the resource was updated and recorded at
      partner service.
    userLabelSet: User-provided labels associated with the resource
    userLabels: User-provided labels, represented as a dictionary where each
      label is a single key value pair.
  """

    class CurrentStateValueValuesEnum(_messages.Enum):
        """Current state of the instance.

    Values:
      STATE_UNSPECIFIED: <no description>
      HEALTHY: The instance is running.
      UNHEALTHY: Instance being created, updated, deleted or under maintenance
      SUSPENDED: When instance is suspended
      DELETED: Instance is deleted.
      STATE_OTHER: For rest of the other category
    """
        STATE_UNSPECIFIED = 0
        HEALTHY = 1
        UNHEALTHY = 2
        SUSPENDED = 3
        DELETED = 4
        STATE_OTHER = 5

    class ExpectedStateValueValuesEnum(_messages.Enum):
        """The state that the instance is expected to be in. For example, an
    instance state can transition to UNHEALTHY due to wrong patch update,
    while the expected state will remain at the HEALTHY.

    Values:
      STATE_UNSPECIFIED: <no description>
      HEALTHY: The instance is running.
      UNHEALTHY: Instance being created, updated, deleted or under maintenance
      SUSPENDED: When instance is suspended
      DELETED: Instance is deleted.
      STATE_OTHER: For rest of the other category
    """
        STATE_UNSPECIFIED = 0
        HEALTHY = 1
        UNHEALTHY = 2
        SUSPENDED = 3
        DELETED = 4
        STATE_OTHER = 5

    class InstanceTypeValueValuesEnum(_messages.Enum):
        """The type of the instance. Specified at creation time.

    Values:
      INSTANCE_TYPE_UNSPECIFIED: <no description>
      SUB_RESOURCE_TYPE_UNSPECIFIED: For rest of the other categories.
      PRIMARY: A regular primary database instance.
      SECONDARY: A cluster or an instance acting as a secondary.
      READ_REPLICA: An instance acting as a read-replica.
      OTHER: For rest of the other categories.
      SUB_RESOURCE_TYPE_PRIMARY: A regular primary database instance.
      SUB_RESOURCE_TYPE_SECONDARY: A cluster or an instance acting as a
        secondary.
      SUB_RESOURCE_TYPE_READ_REPLICA: An instance acting as a read-replica.
      SUB_RESOURCE_TYPE_OTHER: For rest of the other categories.
    """
        INSTANCE_TYPE_UNSPECIFIED = 0
        SUB_RESOURCE_TYPE_UNSPECIFIED = 1
        PRIMARY = 2
        SECONDARY = 3
        READ_REPLICA = 4
        OTHER = 5
        SUB_RESOURCE_TYPE_PRIMARY = 6
        SUB_RESOURCE_TYPE_SECONDARY = 7
        SUB_RESOURCE_TYPE_READ_REPLICA = 8
        SUB_RESOURCE_TYPE_OTHER = 9

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UserLabelsValue(_messages.Message):
        """User-provided labels, represented as a dictionary where each label is
    a single key value pair.

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
    availabilityConfiguration = _messages.MessageField('AvailabilityConfiguration', 1)
    backupConfiguration = _messages.MessageField('BackupConfiguration', 2)
    backupRun = _messages.MessageField('BackupRun', 3)
    creationTime = _messages.StringField(4)
    currentState = _messages.EnumField('CurrentStateValueValuesEnum', 5)
    customMetadata = _messages.MessageField('CustomMetadataData', 6)
    entitlements = _messages.MessageField('Entitlement', 7, repeated=True)
    expectedState = _messages.EnumField('ExpectedStateValueValuesEnum', 8)
    id = _messages.MessageField('DatabaseResourceId', 9)
    instanceType = _messages.EnumField('InstanceTypeValueValuesEnum', 10)
    location = _messages.StringField(11)
    primaryResourceId = _messages.MessageField('DatabaseResourceId', 12)
    product = _messages.MessageField('Product', 13)
    resourceContainer = _messages.StringField(14)
    resourceName = _messages.StringField(15)
    updationTime = _messages.StringField(16)
    userLabelSet = _messages.MessageField('UserLabels', 17)
    userLabels = _messages.MessageField('UserLabelsValue', 18)