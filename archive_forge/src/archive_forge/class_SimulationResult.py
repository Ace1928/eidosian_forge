from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SimulationResult(_messages.Message):
    """SimulationResult provides the simulation result for corresponding
  simulation resource.

  Enums:
    OperationStateValueValuesEnum: Specify the Simulation state.

  Fields:
    connectivityTestSimulationResult: Connectivity test simulation results
      showing diff for each test with pagination.
    executionDuration: Simulation execution duration including a start time
      and an end time.
    operationState: Specify the Simulation state.
    shadowedFirewallSimulationResult: Shadowed firewall simulation results
      showing diff with pagination.
    snapshotTime: Timestamp for data model snapshot used for simulation.
  """

    class OperationStateValueValuesEnum(_messages.Enum):
        """Specify the Simulation state.

    Values:
      OPERATION_STATE_UNSPECIFIED: Default operation state.
      INITIALIZED: Indicates simulation was just initialized.
      SUCCEEDED: Indicates simulation succeeded.
      FAILED: Indicates simulation failed.
      ABORTED: Indicates simulation was aborted.
    """
        OPERATION_STATE_UNSPECIFIED = 0
        INITIALIZED = 1
        SUCCEEDED = 2
        FAILED = 3
        ABORTED = 4
    connectivityTestSimulationResult = _messages.MessageField('ConnectivityTestSimulationResult', 1, repeated=True)
    executionDuration = _messages.StringField(2)
    operationState = _messages.EnumField('OperationStateValueValuesEnum', 3)
    shadowedFirewallSimulationResult = _messages.MessageField('ShadowedFirewallSimulationResult', 4, repeated=True)
    snapshotTime = _messages.StringField(5)