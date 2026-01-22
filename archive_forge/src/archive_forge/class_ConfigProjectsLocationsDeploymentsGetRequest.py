from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigProjectsLocationsDeploymentsGetRequest(_messages.Message):
    """A ConfigProjectsLocationsDeploymentsGetRequest object.

  Fields:
    name: Required. The name of the deployment. Format:
      'projects/{project_id}/locations/{location}/deployments/{deployment}'.
  """
    name = _messages.StringField(1, required=True)