from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Scheduling(_messages.Message):
    """All parameters related to queuing and scheduling of custom jobs.

  Enums:
    StrategyValueValuesEnum: Optional. This determines which type of
      scheduling strategy to use.

  Fields:
    disableRetries: Optional. Indicates if the job should retry for internal
      errors after the job starts running. If true, overrides
      `Scheduling.restart_job_on_worker_restart` to false.
    maxWaitDuration: Optional. This is the maximum duration that a job will
      wait for the requested resources to be provisioned. If set to 0, the job
      will wait indefinitely. The default is 30 minutes.
    restartJobOnWorkerRestart: Restarts the entire CustomJob if a worker gets
      restarted. This feature can be used by distributed training jobs that
      are not resilient to workers leaving and joining a job.
    strategy: Optional. This determines which type of scheduling strategy to
      use.
    timeout: The maximum job running time. The default is 7 days.
  """

    class StrategyValueValuesEnum(_messages.Enum):
        """Optional. This determines which type of scheduling strategy to use.

    Values:
      STRATEGY_UNSPECIFIED: Strategy will default to ON_DEMAND.
      ON_DEMAND: Regular on-demand provisioning strategy.
      LOW_COST: Low cost by making potential use of spot resources.
    """
        STRATEGY_UNSPECIFIED = 0
        ON_DEMAND = 1
        LOW_COST = 2
    disableRetries = _messages.BooleanField(1)
    maxWaitDuration = _messages.StringField(2)
    restartJobOnWorkerRestart = _messages.BooleanField(3)
    strategy = _messages.EnumField('StrategyValueValuesEnum', 4)
    timeout = _messages.StringField(5)