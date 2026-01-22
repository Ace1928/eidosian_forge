from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManager(_messages.Message):
    """Represents a Managed Instance Group resource. An instance group is a
  collection of VM instances that you can manage as a single entity. For more
  information, read Instance groups. For zonal Managed Instance Group, use the
  instanceGroupManagers resource. For regional Managed Instance Group, use the
  regionInstanceGroupManagers resource.

  Enums:
    FailoverActionValueValuesEnum: The action to perform in case of zone
      failure. Only one value is supported, NO_FAILOVER. The default is
      NO_FAILOVER.
    ListManagedInstancesResultsValueValuesEnum: Pagination behavior of the
      listManagedInstances API method for this managed instance group.

  Fields:
    allInstancesConfig: Specifies configuration that overrides the instance
      template configuration for the group.
    autoHealingPolicies: The autohealing policy for this managed instance
      group. You can specify only one value.
    baseInstanceName: The base instance name to use for instances in this
      group. The value must be 1-58 characters long. Instances are named by
      appending a hyphen and a random four-character string to the base
      instance name. The base instance name must comply with RFC1035.
    creationTimestamp: [Output Only] The creation timestamp for this managed
      instance group in RFC3339 text format.
    currentActions: [Output Only] The list of instance actions and the number
      of instances in this managed instance group that are scheduled for each
      of those actions.
    description: An optional description of this resource.
    distributionPolicy: Policy specifying the intended distribution of managed
      instances across zones in a regional managed instance group.
    failoverAction: The action to perform in case of zone failure. Only one
      value is supported, NO_FAILOVER. The default is NO_FAILOVER.
    fingerprint: Fingerprint of this resource. This field may be used in
      optimistic locking. It will be ignored when inserting an
      InstanceGroupManager. An up-to-date fingerprint must be provided in
      order to update the InstanceGroupManager, otherwise the request will
      fail with error 412 conditionNotMet. To see the latest fingerprint, make
      a get() request to retrieve an InstanceGroupManager.
    id: [Output Only] A unique identifier for this resource type. The server
      generates this identifier.
    instanceFlexibilityPolicy: Instance flexibility allowing MIG to create VMs
      from multiple types of machines. Instance flexibility configuration on
      MIG overrides instance template configuration.
    instanceGroup: [Output Only] The URL of the Instance Group resource.
    instanceLifecyclePolicy: The repair policy for this managed instance
      group.
    instanceTemplate: The URL of the instance template that is specified for
      this managed instance group. The group uses this template to create all
      new instances in the managed instance group. The templates for existing
      instances in the group do not change unless you run recreateInstances,
      run applyUpdatesToInstances, or set the group's updatePolicy.type to
      PROACTIVE.
    kind: [Output Only] The resource type, which is always
      compute#instanceGroupManager for managed instance groups.
    listManagedInstancesResults: Pagination behavior of the
      listManagedInstances API method for this managed instance group.
    name: The name of the managed instance group. The name must be 1-63
      characters long, and comply with RFC1035.
    namedPorts: Named ports configured for the Instance Groups complementary
      to this Instance Group Manager.
    params: Input only. Additional params passed with the request, but not
      persisted as part of resource payload.
    region: [Output Only] The URL of the region where the managed instance
      group resides (for regional resources).
    selfLink: [Output Only] The URL for this managed instance group. The
      server defines this URL.
    serviceAccount: The service account to be used as credentials for all
      operations performed by the managed instance group on instances. The
      service accounts needs all permissions required to create and delete
      instances. By default, the service account
      {projectNumber}@cloudservices.gserviceaccount.com is used.
    standbyPolicy: Standby policy for stopped and suspended instances.
    statefulPolicy: Stateful configuration for this Instanced Group Manager
    status: [Output Only] The status of this managed instance group.
    targetPools: The URLs for all TargetPool resources to which instances in
      the instanceGroup field are added. The target pools automatically apply
      to all of the instances in the managed instance group.
    targetSize: The target number of running instances for this managed
      instance group. You can reduce this number by using the
      instanceGroupManager deleteInstances or abandonInstances methods.
      Resizing the group also changes this number.
    targetStoppedSize: The target number of stopped instances for this managed
      instance group. This number changes when you: - Stop instance using the
      stopInstances method or start instances using the startInstances method.
      - Manually change the targetStoppedSize using the update method.
    targetSuspendedSize: The target number of suspended instances for this
      managed instance group. This number changes when you: - Suspend instance
      using the suspendInstances method or resume instances using the
      resumeInstances method. - Manually change the targetSuspendedSize using
      the update method.
    updatePolicy: The update policy for this managed instance group.
    versions: Specifies the instance templates used by this managed instance
      group to create instances. Each version is defined by an
      instanceTemplate and a name. Every version can appear at most once per
      instance group. This field overrides the top-level instanceTemplate
      field. Read more about the relationships between these fields. Exactly
      one version must leave the targetSize field unset. That version will be
      applied to all remaining instances. For more information, read about
      canary updates.
    zone: [Output Only] The URL of a zone where the managed instance group is
      located (for zonal resources).
  """

    class FailoverActionValueValuesEnum(_messages.Enum):
        """The action to perform in case of zone failure. Only one value is
    supported, NO_FAILOVER. The default is NO_FAILOVER.

    Values:
      NO_FAILOVER: <no description>
      UNKNOWN: <no description>
    """
        NO_FAILOVER = 0
        UNKNOWN = 1

    class ListManagedInstancesResultsValueValuesEnum(_messages.Enum):
        """Pagination behavior of the listManagedInstances API method for this
    managed instance group.

    Values:
      PAGELESS: (Default) Pagination is disabled for the group's
        listManagedInstances API method. maxResults and pageToken query
        parameters are ignored and all instances are returned in a single
        response.
      PAGINATED: Pagination is enabled for the group's listManagedInstances
        API method. maxResults and pageToken query parameters are respected.
    """
        PAGELESS = 0
        PAGINATED = 1
    allInstancesConfig = _messages.MessageField('InstanceGroupManagerAllInstancesConfig', 1)
    autoHealingPolicies = _messages.MessageField('InstanceGroupManagerAutoHealingPolicy', 2, repeated=True)
    baseInstanceName = _messages.StringField(3)
    creationTimestamp = _messages.StringField(4)
    currentActions = _messages.MessageField('InstanceGroupManagerActionsSummary', 5)
    description = _messages.StringField(6)
    distributionPolicy = _messages.MessageField('DistributionPolicy', 7)
    failoverAction = _messages.EnumField('FailoverActionValueValuesEnum', 8)
    fingerprint = _messages.BytesField(9)
    id = _messages.IntegerField(10, variant=_messages.Variant.UINT64)
    instanceFlexibilityPolicy = _messages.MessageField('InstanceGroupManagerInstanceFlexibilityPolicy', 11)
    instanceGroup = _messages.StringField(12)
    instanceLifecyclePolicy = _messages.MessageField('InstanceGroupManagerInstanceLifecyclePolicy', 13)
    instanceTemplate = _messages.StringField(14)
    kind = _messages.StringField(15, default='compute#instanceGroupManager')
    listManagedInstancesResults = _messages.EnumField('ListManagedInstancesResultsValueValuesEnum', 16)
    name = _messages.StringField(17)
    namedPorts = _messages.MessageField('NamedPort', 18, repeated=True)
    params = _messages.MessageField('InstanceGroupManagerParams', 19)
    region = _messages.StringField(20)
    selfLink = _messages.StringField(21)
    serviceAccount = _messages.StringField(22)
    standbyPolicy = _messages.MessageField('InstanceGroupManagerStandbyPolicy', 23)
    statefulPolicy = _messages.MessageField('StatefulPolicy', 24)
    status = _messages.MessageField('InstanceGroupManagerStatus', 25)
    targetPools = _messages.StringField(26, repeated=True)
    targetSize = _messages.IntegerField(27, variant=_messages.Variant.INT32)
    targetStoppedSize = _messages.IntegerField(28, variant=_messages.Variant.INT32)
    targetSuspendedSize = _messages.IntegerField(29, variant=_messages.Variant.INT32)
    updatePolicy = _messages.MessageField('InstanceGroupManagerUpdatePolicy', 30)
    versions = _messages.MessageField('InstanceGroupManagerVersion', 31, repeated=True)
    zone = _messages.StringField(32)