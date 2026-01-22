from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2SecurityPosture(_messages.Message):
    """Represents a posture that is deployed on Google Cloud by the Security
  Command Center Posture Management service. A posture contains one or more
  policy sets. A policy set is a group of policies that enforce a set of
  security rules on Google Cloud.

  Fields:
    changedPolicy: The name of the updated policy, for example,
      `projects/{project_id}/policies/{constraint_name}`.
    name: Name of the posture, for example, `CIS-Posture`.
    policy: The ID of the updated policy, for example, `compute-policy-1`.
    policyDriftDetails: The details about a change in an updated policy that
      violates the deployed posture.
    policySet: The name of the updated policy set, for example, `cis-
      policyset`.
    postureDeployment: The name of the posture deployment, for example,
      `organizations/{org_id}/posturedeployments/{posture_deployment_id}`.
    postureDeploymentResource: The project, folder, or organization on which
      the posture is deployed, for example, `projects/{project_number}`.
    revisionId: The version of the posture, for example, `c7cfa2a8`.
  """
    changedPolicy = _messages.StringField(1)
    name = _messages.StringField(2)
    policy = _messages.StringField(3)
    policyDriftDetails = _messages.MessageField('GoogleCloudSecuritycenterV2PolicyDriftDetails', 4, repeated=True)
    policySet = _messages.StringField(5)
    postureDeployment = _messages.StringField(6)
    postureDeploymentResource = _messages.StringField(7)
    revisionId = _messages.StringField(8)