from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerUpdatePolicy(_messages.Message):
    """A InstanceGroupManagerUpdatePolicy object.

  Enums:
    InstanceRedistributionTypeValueValuesEnum: The instance redistribution
      policy for regional managed instance groups. Valid values are: -
      PROACTIVE (default): The group attempts to maintain an even distribution
      of VM instances across zones in the region. - NONE: For non-autoscaled
      groups, proactive redistribution is disabled.
    MinimalActionValueValuesEnum: Minimal action to be taken on an instance.
      Use this option to minimize disruption as much as possible or to apply a
      more disruptive action than is necessary. - To limit disruption as much
      as possible, set the minimal action to REFRESH. If your update requires
      a more disruptive action, Compute Engine performs the necessary action
      to execute the update. - To apply a more disruptive action than is
      strictly necessary, set the minimal action to RESTART or REPLACE. For
      example, Compute Engine does not need to restart a VM to change its
      metadata. But if your application reads instance metadata only when a VM
      is restarted, you can set the minimal action to RESTART in order to pick
      up metadata changes.
    MostDisruptiveAllowedActionValueValuesEnum: Most disruptive action that is
      allowed to be taken on an instance. You can specify either NONE to
      forbid any actions, REFRESH to avoid restarting the VM and to limit
      disruption as much as possible. RESTART to allow actions that can be
      applied without instance replacing or REPLACE to allow all possible
      actions. If the Updater determines that the minimal update action needed
      is more disruptive than most disruptive allowed action you specify it
      will not perform the update at all.
    ReplacementMethodValueValuesEnum: What action should be used to replace
      instances. See minimal_action.REPLACE
    TypeValueValuesEnum: The type of update process. You can specify either
      PROACTIVE so that the MIG automatically updates VMs to the latest
      configurations or OPPORTUNISTIC so that you can select the VMs that you
      want to update.

  Fields:
    instanceRedistributionType: The instance redistribution policy for
      regional managed instance groups. Valid values are: - PROACTIVE
      (default): The group attempts to maintain an even distribution of VM
      instances across zones in the region. - NONE: For non-autoscaled groups,
      proactive redistribution is disabled.
    maxSurge: The maximum number of instances that can be created above the
      specified targetSize during the update process. This value can be either
      a fixed number or, if the group has 10 or more instances, a percentage.
      If you set a percentage, the number of instances is rounded if
      necessary. The default value for maxSurge is a fixed value equal to the
      number of zones in which the managed instance group operates. At least
      one of either maxSurge or maxUnavailable must be greater than 0. Learn
      more about maxSurge.
    maxUnavailable: The maximum number of instances that can be unavailable
      during the update process. An instance is considered available if all of
      the following conditions are satisfied: - The instance's status is
      RUNNING. - If there is a health check on the instance group, the
      instance's health check status must be HEALTHY at least once. If there
      is no health check on the group, then the instance only needs to have a
      status of RUNNING to be considered available. This value can be either a
      fixed number or, if the group has 10 or more instances, a percentage. If
      you set a percentage, the number of instances is rounded if necessary.
      The default value for maxUnavailable is a fixed value equal to the
      number of zones in which the managed instance group operates. At least
      one of either maxSurge or maxUnavailable must be greater than 0. Learn
      more about maxUnavailable.
    minReadySec: Minimum number of seconds to wait for after a newly created
      instance becomes available. This value must be from range [0, 3600].
    minimalAction: Minimal action to be taken on an instance. Use this option
      to minimize disruption as much as possible or to apply a more disruptive
      action than is necessary. - To limit disruption as much as possible, set
      the minimal action to REFRESH. If your update requires a more disruptive
      action, Compute Engine performs the necessary action to execute the
      update. - To apply a more disruptive action than is strictly necessary,
      set the minimal action to RESTART or REPLACE. For example, Compute
      Engine does not need to restart a VM to change its metadata. But if your
      application reads instance metadata only when a VM is restarted, you can
      set the minimal action to RESTART in order to pick up metadata changes.
    mostDisruptiveAllowedAction: Most disruptive action that is allowed to be
      taken on an instance. You can specify either NONE to forbid any actions,
      REFRESH to avoid restarting the VM and to limit disruption as much as
      possible. RESTART to allow actions that can be applied without instance
      replacing or REPLACE to allow all possible actions. If the Updater
      determines that the minimal update action needed is more disruptive than
      most disruptive allowed action you specify it will not perform the
      update at all.
    replacementMethod: What action should be used to replace instances. See
      minimal_action.REPLACE
    type: The type of update process. You can specify either PROACTIVE so that
      the MIG automatically updates VMs to the latest configurations or
      OPPORTUNISTIC so that you can select the VMs that you want to update.
  """

    class InstanceRedistributionTypeValueValuesEnum(_messages.Enum):
        """The instance redistribution policy for regional managed instance
    groups. Valid values are: - PROACTIVE (default): The group attempts to
    maintain an even distribution of VM instances across zones in the region.
    - NONE: For non-autoscaled groups, proactive redistribution is disabled.

    Values:
      NONE: No action is being proactively performed in order to bring this
        IGM to its target instance distribution.
      PROACTIVE: This IGM will actively converge to its target instance
        distribution.
    """
        NONE = 0
        PROACTIVE = 1

    class MinimalActionValueValuesEnum(_messages.Enum):
        """Minimal action to be taken on an instance. Use this option to minimize
    disruption as much as possible or to apply a more disruptive action than
    is necessary. - To limit disruption as much as possible, set the minimal
    action to REFRESH. If your update requires a more disruptive action,
    Compute Engine performs the necessary action to execute the update. - To
    apply a more disruptive action than is strictly necessary, set the minimal
    action to RESTART or REPLACE. For example, Compute Engine does not need to
    restart a VM to change its metadata. But if your application reads
    instance metadata only when a VM is restarted, you can set the minimal
    action to RESTART in order to pick up metadata changes.

    Values:
      NONE: Do not perform any action.
      REFRESH: Do not stop the instance.
      REPLACE: (Default.) Replace the instance according to the replacement
        method option.
      RESTART: Stop the instance and start it again.
    """
        NONE = 0
        REFRESH = 1
        REPLACE = 2
        RESTART = 3

    class MostDisruptiveAllowedActionValueValuesEnum(_messages.Enum):
        """Most disruptive action that is allowed to be taken on an instance. You
    can specify either NONE to forbid any actions, REFRESH to avoid restarting
    the VM and to limit disruption as much as possible. RESTART to allow
    actions that can be applied without instance replacing or REPLACE to allow
    all possible actions. If the Updater determines that the minimal update
    action needed is more disruptive than most disruptive allowed action you
    specify it will not perform the update at all.

    Values:
      NONE: Do not perform any action.
      REFRESH: Do not stop the instance.
      REPLACE: (Default.) Replace the instance according to the replacement
        method option.
      RESTART: Stop the instance and start it again.
    """
        NONE = 0
        REFRESH = 1
        REPLACE = 2
        RESTART = 3

    class ReplacementMethodValueValuesEnum(_messages.Enum):
        """What action should be used to replace instances. See
    minimal_action.REPLACE

    Values:
      RECREATE: Instances will be recreated (with the same name)
      SUBSTITUTE: Default option: instances will be deleted and created (with
        a new name)
    """
        RECREATE = 0
        SUBSTITUTE = 1

    class TypeValueValuesEnum(_messages.Enum):
        """The type of update process. You can specify either PROACTIVE so that
    the MIG automatically updates VMs to the latest configurations or
    OPPORTUNISTIC so that you can select the VMs that you want to update.

    Values:
      OPPORTUNISTIC: MIG will apply new configurations to existing VMs only
        when you selectively target specific or all VMs to be updated.
      PROACTIVE: MIG will automatically apply new configurations to all or a
        subset of existing VMs and also to new VMs that are added to the
        group.
    """
        OPPORTUNISTIC = 0
        PROACTIVE = 1
    instanceRedistributionType = _messages.EnumField('InstanceRedistributionTypeValueValuesEnum', 1)
    maxSurge = _messages.MessageField('FixedOrPercent', 2)
    maxUnavailable = _messages.MessageField('FixedOrPercent', 3)
    minReadySec = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    minimalAction = _messages.EnumField('MinimalActionValueValuesEnum', 5)
    mostDisruptiveAllowedAction = _messages.EnumField('MostDisruptiveAllowedActionValueValuesEnum', 6)
    replacementMethod = _messages.EnumField('ReplacementMethodValueValuesEnum', 7)
    type = _messages.EnumField('TypeValueValuesEnum', 8)