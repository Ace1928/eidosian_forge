from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigProjectsLocationsDeploymentsExportStateRequest(_messages.Message):
    """A ConfigProjectsLocationsDeploymentsExportStateRequest object.

  Fields:
    exportDeploymentStatefileRequest: A ExportDeploymentStatefileRequest
      resource to be passed as the request body.
    parent: Required. The parent in whose context the statefile is listed. The
      parent value is in the format:
      'projects/{project_id}/locations/{location}/deployments/{deployment}'.
  """
    exportDeploymentStatefileRequest = _messages.MessageField('ExportDeploymentStatefileRequest', 1)
    parent = _messages.StringField(2, required=True)