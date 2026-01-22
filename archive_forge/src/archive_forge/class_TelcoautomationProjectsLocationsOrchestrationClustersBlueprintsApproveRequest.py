from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsApproveRequest(_messages.Message):
    """A TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsApprove
  Request object.

  Fields:
    approveBlueprintRequest: A ApproveBlueprintRequest resource to be passed
      as the request body.
    name: Required. The name of the blueprint to approve. The blueprint must
      be in Proposed state. A new revision is committed on approval.
  """
    approveBlueprintRequest = _messages.MessageField('ApproveBlueprintRequest', 1)
    name = _messages.StringField(2, required=True)