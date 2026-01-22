from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerActionsSummary(_messages.Message):
    """A InstanceGroupManagerActionsSummary object.

  Fields:
    abandoning: [Output Only] The total number of instances in the managed
      instance group that are scheduled to be abandoned. Abandoning an
      instance removes it from the managed instance group without deleting it.
    creating: [Output Only] The number of instances in the managed instance
      group that are scheduled to be created or are currently being created.
      If the group fails to create any of these instances, it tries again
      until it creates the instance successfully. If you have disabled
      creation retries, this field will not be populated; instead, the
      creatingWithoutRetries field will be populated.
    creatingWithoutRetries: [Output Only] The number of instances that the
      managed instance group will attempt to create. The group attempts to
      create each instance only once. If the group fails to create any of
      these instances, it decreases the group's targetSize value accordingly.
    deleting: [Output Only] The number of instances in the managed instance
      group that are scheduled to be deleted or are currently being deleted.
    none: [Output Only] The number of instances in the managed instance group
      that are running and have no scheduled actions.
    recreating: [Output Only] The number of instances in the managed instance
      group that are scheduled to be recreated or are currently being being
      recreated. Recreating an instance deletes the existing root persistent
      disk and creates a new disk from the image that is defined in the
      instance template.
    refreshing: [Output Only] The number of instances in the managed instance
      group that are being reconfigured with properties that do not require a
      restart or a recreate action. For example, setting or removing target
      pools for the instance.
    restarting: [Output Only] The number of instances in the managed instance
      group that are scheduled to be restarted or are currently being
      restarted.
    resuming: [Output Only] The number of instances in the managed instance
      group that are scheduled to be resumed or are currently being resumed.
    starting: [Output Only] The number of instances in the managed instance
      group that are scheduled to be started or are currently being started.
    stopping: [Output Only] The number of instances in the managed instance
      group that are scheduled to be stopped or are currently being stopped.
    suspending: [Output Only] The number of instances in the managed instance
      group that are scheduled to be suspended or are currently being
      suspended.
    verifying: [Output Only] The number of instances in the managed instance
      group that are being verified. See the managedInstances[].currentAction
      property in the listManagedInstances method documentation.
  """
    abandoning = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    creating = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    creatingWithoutRetries = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    deleting = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    none = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    recreating = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    refreshing = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    restarting = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    resuming = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    starting = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    stopping = _messages.IntegerField(11, variant=_messages.Variant.INT32)
    suspending = _messages.IntegerField(12, variant=_messages.Variant.INT32)
    verifying = _messages.IntegerField(13, variant=_messages.Variant.INT32)