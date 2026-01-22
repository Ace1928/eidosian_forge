from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedInstance(_messages.Message):
    """A Managed Instance resource.

  Enums:
    CurrentActionValueValuesEnum: [Output Only] The current action that the
      managed instance group has scheduled for the instance. Possible values:
      - NONE The instance is running, and the managed instance group does not
      have any scheduled actions for this instance. - CREATING The managed
      instance group is creating this instance. If the group fails to create
      this instance, it will try again until it is successful. -
      CREATING_WITHOUT_RETRIES The managed instance group is attempting to
      create this instance only once. If the group fails to create this
      instance, it does not try again and the group's targetSize value is
      decreased instead. - RECREATING The managed instance group is recreating
      this instance. - DELETING The managed instance group is permanently
      deleting this instance. - ABANDONING The managed instance group is
      abandoning this instance. The instance will be removed from the instance
      group and from any target pools that are associated with this group. -
      RESTARTING The managed instance group is restarting the instance. -
      REFRESHING The managed instance group is applying configuration changes
      to the instance without stopping it. For example, the group can update
      the target pool list for an instance without stopping that instance. -
      VERIFYING The managed instance group has created the instance and it is
      in the process of being verified.
    InstanceStatusValueValuesEnum: [Output Only] The status of the instance.
      This field is empty when the instance does not exist.
    TargetStatusValueValuesEnum: [Output Only] The eventual status of the
      instance. The instance group manager will not be identified as stable
      till each managed instance reaches its targetStatus.

  Fields:
    allInstancesConfig: [Output Only] Current all-instances configuration
      revision applied to this instance.
    currentAction: [Output Only] The current action that the managed instance
      group has scheduled for the instance. Possible values: - NONE The
      instance is running, and the managed instance group does not have any
      scheduled actions for this instance. - CREATING The managed instance
      group is creating this instance. If the group fails to create this
      instance, it will try again until it is successful. -
      CREATING_WITHOUT_RETRIES The managed instance group is attempting to
      create this instance only once. If the group fails to create this
      instance, it does not try again and the group's targetSize value is
      decreased instead. - RECREATING The managed instance group is recreating
      this instance. - DELETING The managed instance group is permanently
      deleting this instance. - ABANDONING The managed instance group is
      abandoning this instance. The instance will be removed from the instance
      group and from any target pools that are associated with this group. -
      RESTARTING The managed instance group is restarting the instance. -
      REFRESHING The managed instance group is applying configuration changes
      to the instance without stopping it. For example, the group can update
      the target pool list for an instance without stopping that instance. -
      VERIFYING The managed instance group has created the instance and it is
      in the process of being verified.
    id: [Output only] The unique identifier for this resource. This field is
      empty when instance does not exist.
    instance: [Output Only] The URL of the instance. The URL can exist even if
      the instance has not yet been created.
    instanceFlexibilityOverride: [Output Only] The overrides to instance
      properties resulting from InstanceFlexibilityPolicy.
    instanceHealth: [Output Only] Health state of the instance per health-
      check.
    instanceStatus: [Output Only] The status of the instance. This field is
      empty when the instance does not exist.
    lastAttempt: [Output Only] Information about the last attempt to create or
      delete the instance.
    name: [Output Only] The name of the instance. The name always exists even
      if the instance has not yet been created.
    preservedStateFromConfig: [Output Only] Preserved state applied from per-
      instance config for this instance.
    preservedStateFromPolicy: [Output Only] Preserved state generated based on
      stateful policy for this instance.
    propertiesFromFlexibilityPolicy: [Output Only] Instance properties
      selected for this instance resulting from InstanceFlexibilityPolicy.
    targetStatus: [Output Only] The eventual status of the instance. The
      instance group manager will not be identified as stable till each
      managed instance reaches its targetStatus.
    version: [Output Only] Intended version of this instance.
  """

    class CurrentActionValueValuesEnum(_messages.Enum):
        """[Output Only] The current action that the managed instance group has
    scheduled for the instance. Possible values: - NONE The instance is
    running, and the managed instance group does not have any scheduled
    actions for this instance. - CREATING The managed instance group is
    creating this instance. If the group fails to create this instance, it
    will try again until it is successful. - CREATING_WITHOUT_RETRIES The
    managed instance group is attempting to create this instance only once. If
    the group fails to create this instance, it does not try again and the
    group's targetSize value is decreased instead. - RECREATING The managed
    instance group is recreating this instance. - DELETING The managed
    instance group is permanently deleting this instance. - ABANDONING The
    managed instance group is abandoning this instance. The instance will be
    removed from the instance group and from any target pools that are
    associated with this group. - RESTARTING The managed instance group is
    restarting the instance. - REFRESHING The managed instance group is
    applying configuration changes to the instance without stopping it. For
    example, the group can update the target pool list for an instance without
    stopping that instance. - VERIFYING The managed instance group has created
    the instance and it is in the process of being verified.

    Values:
      ABANDONING: The managed instance group is abandoning this instance. The
        instance will be removed from the instance group and from any target
        pools that are associated with this group.
      CREATING: The managed instance group is creating this instance. If the
        group fails to create this instance, it will try again until it is
        successful.
      CREATING_WITHOUT_RETRIES: The managed instance group is attempting to
        create this instance only once. If the group fails to create this
        instance, it does not try again and the group's targetSize value is
        decreased.
      DELETING: The managed instance group is permanently deleting this
        instance.
      NONE: The managed instance group has not scheduled any actions for this
        instance.
      RECREATING: The managed instance group is recreating this instance.
      REFRESHING: The managed instance group is applying configuration changes
        to the instance without stopping it. For example, the group can update
        the target pool list for an instance without stopping that instance.
      RESTARTING: The managed instance group is restarting this instance.
      RESUMING: The managed instance group is resuming this instance.
      STARTING: The managed instance group is starting this instance.
      STOPPING: The managed instance group is stopping this instance.
      SUSPENDING: The managed instance group is suspending this instance.
      VERIFYING: The managed instance group is verifying this already created
        instance. Verification happens every time the instance is (re)created
        or restarted and consists of: 1. Waiting until health check specified
        as part of this managed instance group's autohealing policy reports
        HEALTHY. Note: Applies only if autohealing policy has a health check
        specified 2. Waiting for addition verification steps performed as
        post-instance creation (subject to future extensions).
    """
        ABANDONING = 0
        CREATING = 1
        CREATING_WITHOUT_RETRIES = 2
        DELETING = 3
        NONE = 4
        RECREATING = 5
        REFRESHING = 6
        RESTARTING = 7
        RESUMING = 8
        STARTING = 9
        STOPPING = 10
        SUSPENDING = 11
        VERIFYING = 12

    class InstanceStatusValueValuesEnum(_messages.Enum):
        """[Output Only] The status of the instance. This field is empty when the
    instance does not exist.

    Values:
      DEPROVISIONING: The instance is halted and we are performing tear down
        tasks like network deprogramming, releasing quota, IP, tearing down
        disks etc.
      PROVISIONING: Resources are being allocated for the instance.
      REPAIRING: The instance is in repair.
      RUNNING: The instance is running.
      STAGING: All required resources have been allocated and the instance is
        being started.
      STOPPED: The instance has stopped successfully.
      STOPPING: The instance is currently stopping (either being deleted or
        killed).
      SUSPENDED: The instance has suspended.
      SUSPENDING: The instance is suspending.
      TERMINATED: The instance has stopped (either by explicit action or
        underlying failure).
    """
        DEPROVISIONING = 0
        PROVISIONING = 1
        REPAIRING = 2
        RUNNING = 3
        STAGING = 4
        STOPPED = 5
        STOPPING = 6
        SUSPENDED = 7
        SUSPENDING = 8
        TERMINATED = 9

    class TargetStatusValueValuesEnum(_messages.Enum):
        """[Output Only] The eventual status of the instance. The instance group
    manager will not be identified as stable till each managed instance
    reaches its targetStatus.

    Values:
      ABANDONED: The managed instance will eventually be ABANDONED, i.e.
        dissociated from the managed instance group.
      DELETED: The managed instance will eventually be DELETED.
      RUNNING: The managed instance will eventually reach status RUNNING.
      STOPPED: The managed instance will eventually reach status TERMINATED.
      SUSPENDED: The managed instance will eventually reach status SUSPENDED.
    """
        ABANDONED = 0
        DELETED = 1
        RUNNING = 2
        STOPPED = 3
        SUSPENDED = 4
    allInstancesConfig = _messages.MessageField('ManagedInstanceAllInstancesConfig', 1)
    currentAction = _messages.EnumField('CurrentActionValueValuesEnum', 2)
    id = _messages.IntegerField(3, variant=_messages.Variant.UINT64)
    instance = _messages.StringField(4)
    instanceFlexibilityOverride = _messages.MessageField('ManagedInstanceInstanceFlexibilityOverride', 5)
    instanceHealth = _messages.MessageField('ManagedInstanceInstanceHealth', 6, repeated=True)
    instanceStatus = _messages.EnumField('InstanceStatusValueValuesEnum', 7)
    lastAttempt = _messages.MessageField('ManagedInstanceLastAttempt', 8)
    name = _messages.StringField(9)
    preservedStateFromConfig = _messages.MessageField('PreservedState', 10)
    preservedStateFromPolicy = _messages.MessageField('PreservedState', 11)
    propertiesFromFlexibilityPolicy = _messages.MessageField('ManagedInstancePropertiesFromFlexibilityPolicy', 12)
    targetStatus = _messages.EnumField('TargetStatusValueValuesEnum', 13)
    version = _messages.MessageField('ManagedInstanceVersion', 14)