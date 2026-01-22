from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureNodePool(_messages.Message):
    """An Anthos node pool running on Azure.

  Enums:
    StateValueValuesEnum: Output only. The current state of the node pool.

  Messages:
    AnnotationsValue: Optional. Annotations on the node pool. This field has
      the same restrictions as Kubernetes annotations. The total size of all
      keys and values combined is limited to 256k. Keys can have 2 segments:
      prefix (optional) and name (required), separated by a slash (/). Prefix
      must be a DNS subdomain. Name must be 63 characters or less, begin and
      end with alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.

  Fields:
    annotations: Optional. Annotations on the node pool. This field has the
      same restrictions as Kubernetes annotations. The total size of all keys
      and values combined is limited to 256k. Keys can have 2 segments: prefix
      (optional) and name (required), separated by a slash (/). Prefix must be
      a DNS subdomain. Name must be 63 characters or less, begin and end with
      alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.
    autoscaling: Required. Autoscaler configuration for this node pool.
    azureAvailabilityZone: Optional. The Azure availability zone of the nodes
      in this nodepool. When unspecified, it defaults to `1`.
    config: Required. The node configuration of the node pool.
    createTime: Output only. The time at which this node pool was created.
    errors: Output only. A set of errors found in the node pool.
    etag: Allows clients to perform consistent read-modify-writes through
      optimistic concurrency control. Can be sent on update and delete
      requests to ensure the client has an up-to-date value before proceeding.
    management: Optional. The Management configuration for this node pool.
    maxPodsConstraint: Required. The constraint on the maximum number of pods
      that can be run simultaneously on a node in the node pool.
    name: The name of this resource. Node pool names are formatted as
      `projects//locations//azureClusters//azureNodePools/`. For more details
      on Google Cloud resource names, see [Resource
      Names](https://cloud.google.com/apis/design/resource_names)
    reconciling: Output only. If set, there are currently pending changes to
      the node pool.
    state: Output only. The current state of the node pool.
    subnetId: Required. The ARM ID of the subnet where the node pool VMs run.
      Make sure it's a subnet under the virtual network in the cluster
      configuration.
    uid: Output only. A globally unique identifier for the node pool.
    updateTime: Output only. The time at which this node pool was last
      updated.
    version: Required. The Kubernetes version (e.g. `1.19.10-gke.1000`)
      running on this node pool.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the node pool.

    Values:
      STATE_UNSPECIFIED: Not set.
      PROVISIONING: The PROVISIONING state indicates the node pool is being
        created.
      RUNNING: The RUNNING state indicates the node pool has been created and
        is fully usable.
      RECONCILING: The RECONCILING state indicates that the node pool is being
        reconciled.
      STOPPING: The STOPPING state indicates the node pool is being deleted.
      ERROR: The ERROR state indicates the node pool is in a broken
        unrecoverable state.
      DEGRADED: The DEGRADED state indicates the node pool requires user
        action to restore full functionality.
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
        """Optional. Annotations on the node pool. This field has the same
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
    autoscaling = _messages.MessageField('GoogleCloudGkemulticloudV1AzureNodePoolAutoscaling', 2)
    azureAvailabilityZone = _messages.StringField(3)
    config = _messages.MessageField('GoogleCloudGkemulticloudV1AzureNodeConfig', 4)
    createTime = _messages.StringField(5)
    errors = _messages.MessageField('GoogleCloudGkemulticloudV1AzureNodePoolError', 6, repeated=True)
    etag = _messages.StringField(7)
    management = _messages.MessageField('GoogleCloudGkemulticloudV1AzureNodeManagement', 8)
    maxPodsConstraint = _messages.MessageField('GoogleCloudGkemulticloudV1MaxPodsConstraint', 9)
    name = _messages.StringField(10)
    reconciling = _messages.BooleanField(11)
    state = _messages.EnumField('StateValueValuesEnum', 12)
    subnetId = _messages.StringField(13)
    uid = _messages.StringField(14)
    updateTime = _messages.StringField(15)
    version = _messages.StringField(16)