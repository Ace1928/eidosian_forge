from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsCreateRequest(_messages.Message):
    """A
  TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsCreateRequest
  object.

  Fields:
    blueprint: A Blueprint resource to be passed as the request body.
    blueprintId: Optional. The name of the blueprint.
    parent: Required. The name of parent resource. Format should be - "project
      s/{project_id}/locations/{location_name}/orchestrationClusters/{orchestr
      ation_cluster}".
  """
    blueprint = _messages.MessageField('Blueprint', 1)
    blueprintId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)