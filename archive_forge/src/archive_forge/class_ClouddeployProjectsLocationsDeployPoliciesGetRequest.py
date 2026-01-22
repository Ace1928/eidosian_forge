from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeployPoliciesGetRequest(_messages.Message):
    """A ClouddeployProjectsLocationsDeployPoliciesGetRequest object.

  Fields:
    name: Required. Name of the `DeployPolicy`. Format must be `projects/{proj
      ect_id}/locations/{location_name}/deployPolicies/{deploy_policy_name}`.
  """
    name = _messages.StringField(1, required=True)