from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HydratedDeployment(_messages.Message):
    """A collection of kubernetes yaml files which are deployed on a Workload
  Cluster. Hydrated Deployments are created by TNA intent based automation.

  Enums:
    StateValueValuesEnum: Output only. State of the hydrated deployment
      (DRAFT, APPLIED).

  Fields:
    files: Optional. File contents of a hydrated deployment. When invoking
      UpdateHydratedBlueprint API, only the modified files should be included
      in this. Files that are not included in the update of a hydrated
      deployment will not be changed.
    name: Output only. The name of the hydrated deployment.
    state: Output only. State of the hydrated deployment (DRAFT, APPLIED).
    workloadCluster: Output only. WorkloadCluster identifies which workload
      cluster will the hydrated deployment will be deployed on.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the hydrated deployment (DRAFT, APPLIED).

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      DRAFT: A hydrated deployment starts in DRAFT state. All edits are made
        in DRAFT state.
      APPLIED: When the edit is applied, the hydrated deployment moves to
        APPLIED state. No changes can be made once a hydrated deployment is
        applied.
    """
        STATE_UNSPECIFIED = 0
        DRAFT = 1
        APPLIED = 2
    files = _messages.MessageField('File', 1, repeated=True)
    name = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    workloadCluster = _messages.StringField(4)