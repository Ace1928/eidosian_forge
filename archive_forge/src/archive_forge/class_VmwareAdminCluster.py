from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareAdminCluster(_messages.Message):
    """Resource that represents a VMware admin cluster.

  Enums:
    StateValueValuesEnum: Output only. The current state of VMware admin
      cluster.

  Messages:
    AnnotationsValue: Annotations on the VMware admin cluster. This field has
      the same restrictions as Kubernetes annotations. The total size of all
      keys and values combined is limited to 256k. Key can have 2 segments:
      prefix (optional) and name (required), separated by a slash (/). Prefix
      must be a DNS subdomain. Name must be 63 characters or less, begin and
      end with alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.

  Fields:
    addonNode: The VMware admin cluster addon node configuration.
    annotations: Annotations on the VMware admin cluster. This field has the
      same restrictions as Kubernetes annotations. The total size of all keys
      and values combined is limited to 256k. Key can have 2 segments: prefix
      (optional) and name (required), separated by a slash (/). Prefix must be
      a DNS subdomain. Name must be 63 characters or less, begin and end with
      alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.
    antiAffinityGroups: The VMware admin cluster anti affinity group
      configuration.
    authorization: The VMware admin cluster authorization configuration.
    autoRepairConfig: The VMware admin cluster auto repair configuration.
    bootstrapClusterMembership: The bootstrap cluster this VMware admin
      cluster belongs to.
    controlPlaneNode: The VMware admin cluster control plane node
      configuration.
    createTime: Output only. The time at which VMware admin cluster was
      created.
    description: A human readable description of this VMware admin cluster.
    endpoint: Output only. The DNS name of VMware admin cluster's API server.
    etag: This checksum is computed by the server based on the value of other
      fields, and may be sent on update and delete requests to ensure the
      client has an up-to-date value before proceeding. Allows clients to
      perform consistent read-modify-writes through optimistic concurrency
      control.
    fleet: Output only. Fleet configuration for the cluster.
    imageType: The OS image type for the VMware admin cluster.
    loadBalancer: The VMware admin cluster load balancer configuration.
    localName: Output only. The object name of the VMware OnPremAdminCluster
      custom resource. This field is used to support conflicting names when
      enrolling existing clusters to the API. When used as a part of cluster
      enrollment, this field will differ from the ID in the resource name. For
      new clusters, this field will match the user provided cluster name and
      be visible in the last component of the resource name. It is not
      modifiable. All users should use this name to access their cluster using
      gkectl or kubectl and should expect to see the local name when viewing
      admin cluster controller logs.
    name: Immutable. The VMware admin cluster resource name.
    networkConfig: The VMware admin cluster network configuration.
    onPremVersion: The Anthos clusters on the VMware version for the admin
      cluster.
    platformConfig: The VMware platform configuration.
    preparedSecrets: Output only. The VMware admin cluster prepared secrets
      configuration. It should always be enabled by the Central API, instead
      of letting users set it.
    reconciling: Output only. If set, there are currently changes in flight to
      the VMware admin cluster.
    state: Output only. The current state of VMware admin cluster.
    status: Output only. ResourceStatus representing detailed cluster state.
    uid: Output only. The unique identifier of the VMware admin cluster.
    updateTime: Output only. The time at which VMware admin cluster was last
      updated.
    validationCheck: Output only. ValidationCheck represents the result of the
      preflight check job.
    vcenter: The VMware admin cluster VCenter configuration.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of VMware admin cluster.

    Values:
      STATE_UNSPECIFIED: Not set.
      PROVISIONING: The PROVISIONING state indicates the cluster is being
        created.
      RUNNING: The RUNNING state indicates the cluster has been created and is
        fully usable.
      RECONCILING: The RECONCILING state indicates that the cluster is being
        updated. It remains available, but potentially with degraded
        performance.
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
        """Annotations on the VMware admin cluster. This field has the same
    restrictions as Kubernetes annotations. The total size of all keys and
    values combined is limited to 256k. Key can have 2 segments: prefix
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
    addonNode = _messages.MessageField('VmwareAdminAddonNodeConfig', 1)
    annotations = _messages.MessageField('AnnotationsValue', 2)
    antiAffinityGroups = _messages.MessageField('VmwareAAGConfig', 3)
    authorization = _messages.MessageField('VmwareAdminAuthorizationConfig', 4)
    autoRepairConfig = _messages.MessageField('VmwareAutoRepairConfig', 5)
    bootstrapClusterMembership = _messages.StringField(6)
    controlPlaneNode = _messages.MessageField('VmwareAdminControlPlaneNodeConfig', 7)
    createTime = _messages.StringField(8)
    description = _messages.StringField(9)
    endpoint = _messages.StringField(10)
    etag = _messages.StringField(11)
    fleet = _messages.MessageField('Fleet', 12)
    imageType = _messages.StringField(13)
    loadBalancer = _messages.MessageField('VmwareAdminLoadBalancerConfig', 14)
    localName = _messages.StringField(15)
    name = _messages.StringField(16)
    networkConfig = _messages.MessageField('VmwareAdminNetworkConfig', 17)
    onPremVersion = _messages.StringField(18)
    platformConfig = _messages.MessageField('VmwarePlatformConfig', 19)
    preparedSecrets = _messages.MessageField('VmwareAdminPreparedSecretsConfig', 20)
    reconciling = _messages.BooleanField(21)
    state = _messages.EnumField('StateValueValuesEnum', 22)
    status = _messages.MessageField('ResourceStatus', 23)
    uid = _messages.StringField(24)
    updateTime = _messages.StringField(25)
    validationCheck = _messages.MessageField('ValidationCheck', 26)
    vcenter = _messages.MessageField('VmwareAdminVCenterConfig', 27)