from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureCluster(_messages.Message):
    """An Anthos cluster running on Azure.

  Enums:
    StateValueValuesEnum: Output only. The current state of the cluster.

  Messages:
    AnnotationsValue: Optional. Annotations on the cluster. This field has the
      same restrictions as Kubernetes annotations. The total size of all keys
      and values combined is limited to 256k. Keys can have 2 segments: prefix
      (optional) and name (required), separated by a slash (/). Prefix must be
      a DNS subdomain. Name must be 63 characters or less, begin and end with
      alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.

  Fields:
    annotations: Optional. Annotations on the cluster. This field has the same
      restrictions as Kubernetes annotations. The total size of all keys and
      values combined is limited to 256k. Keys can have 2 segments: prefix
      (optional) and name (required), separated by a slash (/). Prefix must be
      a DNS subdomain. Name must be 63 characters or less, begin and end with
      alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.
    authorization: Required. Configuration related to the cluster RBAC
      settings.
    azureClient: Optional. Name of the AzureClient that contains
      authentication configuration for how the Anthos Multi-Cloud API connects
      to Azure APIs. Either azure_client or azure_services_authentication
      should be provided. The `AzureClient` resource must reside on the same
      Google Cloud Platform project and region as the `AzureCluster`.
      `AzureClient` names are formatted as
      `projects//locations//azureClients/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
    azureRegion: Required. The Azure region where the cluster runs. Each
      Google Cloud region supports a subset of nearby Azure regions. You can
      call GetAzureServerConfig to list all supported Azure regions within a
      given Google Cloud region.
    azureServicesAuthentication: Optional. Authentication configuration for
      management of Azure resources. Either azure_client or
      azure_services_authentication should be provided.
    clusterCaCertificate: Output only. PEM encoded x509 certificate of the
      cluster root of trust.
    controlPlane: Required. Configuration related to the cluster control
      plane.
    createTime: Output only. The time at which this cluster was created.
    description: Optional. A human readable description of this cluster.
      Cannot be longer than 255 UTF-8 encoded bytes.
    endpoint: Output only. The endpoint of the cluster's API server.
    errors: Output only. A set of errors found in the cluster.
    etag: Allows clients to perform consistent read-modify-writes through
      optimistic concurrency control. Can be sent on update and delete
      requests to ensure the client has an up-to-date value before proceeding.
    fleet: Required. Fleet configuration.
    loggingConfig: Optional. Logging configuration for this cluster.
    managedResources: Output only. Managed Azure resources for this cluster.
    monitoringConfig: Optional. Monitoring configuration for this cluster.
    name: The name of this resource. Cluster names are formatted as
      `projects//locations//azureClusters/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud Platform resource names.
    networking: Required. Cluster-wide networking configuration.
    reconciling: Output only. If set, there are currently changes in flight to
      the cluster.
    resourceGroupId: Required. The ARM ID of the resource group where the
      cluster resources are deployed. For example:
      `/subscriptions//resourceGroups/`
    state: Output only. The current state of the cluster.
    uid: Output only. A globally unique identifier for the cluster.
    updateTime: Output only. The time at which this cluster was last updated.
    workloadIdentityConfig: Output only. Workload Identity settings.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the cluster.

    Values:
      STATE_UNSPECIFIED: Not set.
      PROVISIONING: The PROVISIONING state indicates the cluster is being
        created.
      RUNNING: The RUNNING state indicates the cluster has been created and is
        fully usable.
      RECONCILING: The RECONCILING state indicates that some work is actively
        being done on the cluster, such as upgrading the control plane
        replicas.
      STOPPING: The STOPPING state indicates the cluster is being deleted.
      ERROR: The ERROR state indicates the cluster is in a broken
        unrecoverable state.
      DEGRADED: The DEGRADED state indicates the cluster requires user action
        to restore full functionality.
    """
        STATE_UNSPECIFIED = 0
        PROVISIONING = 1
        RUNNING = 2
        RECONCILING = 3
        STOPPING = 4
        ERROR = 5
        DEGRADED = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. Annotations on the cluster. This field has the same
    restrictions as Kubernetes annotations. The total size of all keys and
    values combined is limited to 256k. Keys can have 2 segments: prefix
    (optional) and name (required), separated by a slash (/). Prefix must be a
    DNS subdomain. Name must be 63 characters or less, begin and end with
    alphanumerics, with dashes (-), underscores (_), dots (.), and
    alphanumerics between.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    authorization = _messages.MessageField('GoogleCloudGkemulticloudV1AzureAuthorization', 2)
    azureClient = _messages.StringField(3)
    azureRegion = _messages.StringField(4)
    azureServicesAuthentication = _messages.MessageField('GoogleCloudGkemulticloudV1AzureServicesAuthentication', 5)
    clusterCaCertificate = _messages.StringField(6)
    controlPlane = _messages.MessageField('GoogleCloudGkemulticloudV1AzureControlPlane', 7)
    createTime = _messages.StringField(8)
    description = _messages.StringField(9)
    endpoint = _messages.StringField(10)
    errors = _messages.MessageField('GoogleCloudGkemulticloudV1AzureClusterError', 11, repeated=True)
    etag = _messages.StringField(12)
    fleet = _messages.MessageField('GoogleCloudGkemulticloudV1Fleet', 13)
    loggingConfig = _messages.MessageField('GoogleCloudGkemulticloudV1LoggingConfig', 14)
    managedResources = _messages.MessageField('GoogleCloudGkemulticloudV1AzureClusterResources', 15)
    monitoringConfig = _messages.MessageField('GoogleCloudGkemulticloudV1MonitoringConfig', 16)
    name = _messages.StringField(17)
    networking = _messages.MessageField('GoogleCloudGkemulticloudV1AzureClusterNetworking', 18)
    reconciling = _messages.BooleanField(19)
    resourceGroupId = _messages.StringField(20)
    state = _messages.EnumField('StateValueValuesEnum', 21)
    uid = _messages.StringField(22)
    updateTime = _messages.StringField(23)
    workloadIdentityConfig = _messages.MessageField('GoogleCloudGkemulticloudV1WorkloadIdentityConfig', 24)