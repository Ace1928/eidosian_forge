from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSaasacceleratorManagementProvidersV1Instance(_messages.Message):
    """Instance represents the interface for SLM services to actuate the state
  of control plane resources. Example Instance in JSON, where consumer-
  project-number=123456, producer-project-id=cloud-sql: ```json Instance: {
  "name": "projects/123456/locations/us-east1/instances/prod-instance",
  "create_time": { "seconds": 1526406431, }, "labels": { "env": "prod", "foo":
  "bar" }, "state": READY, "software_versions": { "software_update": "cloud-
  sql-09-28-2018", }, "maintenance_policy_names": { "UpdatePolicy":
  "projects/123456/locations/us-east1/maintenancePolicies/prod-update-policy",
  } "tenant_project_id": "cloud-sql-test-tenant", "producer_metadata": {
  "cloud-sql-tier": "basic", "cloud-sql-instance-size": "1G", },
  "provisioned_resources": [ { "resource-type": "compute-instance", "resource-
  url": "https://www.googleapis.com/compute/v1/projects/cloud-sql/zones/us-
  east1-b/instances/vm-1", } ], "maintenance_schedules": { "csa_rollout": {
  "start_time": { "seconds": 1526406431, }, "end_time": { "seconds":
  1535406431, }, }, "ncsa_rollout": { "start_time": { "seconds": 1526406431,
  }, "end_time": { "seconds": 1535406431, }, } }, "consumer_defined_name":
  "my-sql-instance1", } ``` LINT.IfChange

  Enums:
    StateValueValuesEnum: Output only. Current lifecycle state of the resource
      (e.g. if it's being created or ready to use).

  Messages:
    LabelsValue: Optional. Resource labels to represent user provided
      metadata. Each label is a key-value pair, where both the key and the
      value are arbitrary strings provided by the user.
    MaintenancePolicyNamesValue: Optional. The MaintenancePolicies that have
      been attached to the instance. The key must be of the type name of the
      oneof policy name defined in MaintenancePolicy, and the referenced
      policy must define the same policy type. For details, please refer to
      go/mr-user-guide. Should not be set if
      maintenance_settings.maintenance_policies is set.
    MaintenanceSchedulesValue: The MaintenanceSchedule contains the scheduling
      information of published maintenance schedule with same key as
      software_versions.
    NotificationParametersValue: Optional. notification_parameter are
      information that service producers may like to include that is not
      relevant to Rollout. This parameter will only be passed to Gamma and
      Cloud Logging for notification/logging purpose.
    ProducerMetadataValue: Output only. Custom string attributes used
      primarily to expose producer-specific information in monitoring
      dashboards. See go/get-instance-metadata.
    SoftwareVersionsValue: Software versions that are used to deploy this
      instance. This can be mutated by rollout services.

  Fields:
    consumerDefinedName: consumer_defined_name is the name of the instance set
      by the service consumers. Generally this is different from the `name`
      field which reperesents the system-assigned id of the instance which the
      service consumers do not recognize. This is a required field for tenants
      onboarding to Maintenance Window notifications (go/slm-rollout-
      maintenance-policies#prerequisites).
    createTime: Output only. Timestamp when the resource was created.
    instanceType: Optional. The instance_type of this instance of format: proj
      ects/{project_number}/locations/{location_id}/instanceTypes/{instance_ty
      pe_id}. Instance Type represents a high-level tier or SKU of the service
      that this instance belong to. When enabled(eg: Maintenance Rollout),
      Rollout uses 'instance_type' along with 'software_versions' to determine
      whether instance needs an update or not.
    labels: Optional. Resource labels to represent user provided metadata.
      Each label is a key-value pair, where both the key and the value are
      arbitrary strings provided by the user.
    maintenancePolicyNames: Optional. The MaintenancePolicies that have been
      attached to the instance. The key must be of the type name of the oneof
      policy name defined in MaintenancePolicy, and the referenced policy must
      define the same policy type. For details, please refer to go/mr-user-
      guide. Should not be set if maintenance_settings.maintenance_policies is
      set.
    maintenanceSchedules: The MaintenanceSchedule contains the scheduling
      information of published maintenance schedule with same key as
      software_versions.
    maintenanceSettings: Optional. The MaintenanceSettings associated with
      instance.
    name: Unique name of the resource. It uses the form: `projects/{project_nu
      mber}/locations/{location_id}/instances/{instance_id}` Note: This name
      is passed, stored and logged across the rollout system. So use of
      consumer project_id or any other consumer PII in the name is strongly
      discouraged for wipeout (go/wipeout) compliance. See
      go/elysium/project_ids#storage-guidance for more details.
    notificationParameters: Optional. notification_parameter are information
      that service producers may like to include that is not relevant to
      Rollout. This parameter will only be passed to Gamma and Cloud Logging
      for notification/logging purpose.
    producerMetadata: Output only. Custom string attributes used primarily to
      expose producer-specific information in monitoring dashboards. See
      go/get-instance-metadata.
    provisionedResources: Output only. The list of data plane resources
      provisioned for this instance, e.g. compute VMs. See go/get-instance-
      metadata.
    slmInstanceTemplate: Link to the SLM instance template. Only populated
      when updating SLM instances via SSA's Actuation service adaptor. Service
      producers with custom control plane (e.g. Cloud SQL) doesn't need to
      populate this field. Instead they should use software_versions.
    sloMetadata: Output only. SLO metadata for instance classification in the
      Standardized dataplane SLO platform. See go/cloud-ssa-standard-slo for
      feature description.
    softwareVersions: Software versions that are used to deploy this instance.
      This can be mutated by rollout services.
    state: Output only. Current lifecycle state of the resource (e.g. if it's
      being created or ready to use).
    tenantProjectId: Output only. ID of the associated GCP tenant project. See
      go/get-instance-metadata.
    updateTime: Output only. Timestamp when the resource was last modified.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current lifecycle state of the resource (e.g. if it's
    being created or ready to use).

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      CREATING: Instance is being created.
      READY: Instance has been created and is ready to use.
      UPDATING: Instance is being updated.
      REPAIRING: Instance is unheathy and under repair.
      DELETING: Instance is being deleted.
      ERROR: Instance encountered an error and is in indeterministic state.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
        UPDATING = 3
        REPAIRING = 4
        DELETING = 5
        ERROR = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Resource labels to represent user provided metadata. Each
    label is a key-value pair, where both the key and the value are arbitrary
    strings provided by the user.

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
    class MaintenancePolicyNamesValue(_messages.Message):
        """Optional. The MaintenancePolicies that have been attached to the
    instance. The key must be of the type name of the oneof policy name
    defined in MaintenancePolicy, and the referenced policy must define the
    same policy type. For details, please refer to go/mr-user-guide. Should
    not be set if maintenance_settings.maintenance_policies is set.

    Messages:
      AdditionalProperty: An additional property for a
        MaintenancePolicyNamesValue object.

    Fields:
      additionalProperties: Additional properties of type
        MaintenancePolicyNamesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MaintenancePolicyNamesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class NotificationParametersValue(_messages.Message):
        """Optional. notification_parameter are information that service
    producers may like to include that is not relevant to Rollout. This
    parameter will only be passed to Gamma and Cloud Logging for
    notification/logging purpose.

    Messages:
      AdditionalProperty: An additional property for a
        NotificationParametersValue object.

    Fields:
      additionalProperties: Additional properties of type
        NotificationParametersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a NotificationParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A
          GoogleCloudSaasacceleratorManagementProvidersV1NotificationParameter
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudSaasacceleratorManagementProvidersV1NotificationParameter', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ProducerMetadataValue(_messages.Message):
        """Output only. Custom string attributes used primarily to expose
    producer-specific information in monitoring dashboards. See go/get-
    instance-metadata.

    Messages:
      AdditionalProperty: An additional property for a ProducerMetadataValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ProducerMetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ProducerMetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SoftwareVersionsValue(_messages.Message):
        """Software versions that are used to deploy this instance. This can be
    mutated by rollout services.

    Messages:
      AdditionalProperty: An additional property for a SoftwareVersionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        SoftwareVersionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SoftwareVersionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    consumerDefinedName = _messages.StringField(1)
    createTime = _messages.StringField(2)
    instanceType = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    maintenancePolicyNames = _messages.MessageField('MaintenancePolicyNamesValue', 5)
    maintenanceSchedules = _messages.MessageField('MaintenanceSchedulesValue', 6)
    maintenanceSettings = _messages.MessageField('GoogleCloudSaasacceleratorManagementProvidersV1MaintenanceSettings', 7)
    name = _messages.StringField(8)
    notificationParameters = _messages.MessageField('NotificationParametersValue', 9)
    producerMetadata = _messages.MessageField('ProducerMetadataValue', 10)
    provisionedResources = _messages.MessageField('GoogleCloudSaasacceleratorManagementProvidersV1ProvisionedResource', 11, repeated=True)
    slmInstanceTemplate = _messages.StringField(12)
    sloMetadata = _messages.MessageField('GoogleCloudSaasacceleratorManagementProvidersV1SloMetadata', 13)
    softwareVersions = _messages.MessageField('SoftwareVersionsValue', 14)
    state = _messages.EnumField('StateValueValuesEnum', 15)
    tenantProjectId = _messages.StringField(16)
    updateTime = _messages.StringField(17)