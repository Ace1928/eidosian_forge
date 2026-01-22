from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaWorkerPool(_messages.Message):
    """A worker pool resource in the Remote Build Execution API.

  Enums:
    StateValueValuesEnum: Output only. State of the worker pool.

  Fields:
    autoscale: The autoscale policy to apply on a pool.
    channel: Channel specifies the release channel of the pool.
    hostOs: HostOS specifies the OS version of the image for the worker VMs.
    name: WorkerPool resource name formatted as:
      `projects/[PROJECT_ID]/instances/[INSTANCE_ID]/workerpools/[POOL_ID]`.
      name should not be populated when creating a worker pool since it is
      provided in the `poolId` field.
    state: Output only. State of the worker pool.
    workerConfig: Specifies the properties, such as machine type and disk
      size, used for creating workers in a worker pool.
    workerCount: The desired number of workers in the worker pool. Must be a
      value between 0 and 15000.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the worker pool.

    Values:
      STATE_UNSPECIFIED: Not a valid state, but the default value of the enum.
      CREATING: The worker pool is in state `CREATING` once `CreateWorkerPool`
        is called and before all requested workers are ready.
      RUNNING: The worker pool is in state `RUNNING` when all its workers are
        ready for use.
      UPDATING: The worker pool is in state `UPDATING` once `UpdateWorkerPool`
        is called and before the new configuration has all the requested
        workers ready for use, and no older configuration has any workers. At
        that point the state transitions to `RUNNING`.
      DELETING: The worker pool is in state `DELETING` once the `Delete`
        method is called and before the deletion completes.
      INACTIVE: The worker pool is in state `INACTIVE` when the instance
        hosting the worker pool in not running.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        RUNNING = 2
        UPDATING = 3
        DELETING = 4
        INACTIVE = 5
    autoscale = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaAutoscale', 1)
    channel = _messages.StringField(2)
    hostOs = _messages.StringField(3)
    name = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    workerConfig = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaWorkerConfig', 6)
    workerCount = _messages.IntegerField(7)