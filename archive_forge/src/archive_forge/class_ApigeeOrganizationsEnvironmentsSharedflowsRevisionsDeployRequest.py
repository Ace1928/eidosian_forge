from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSharedflowsRevisionsDeployRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSharedflowsRevisionsDeployRequest
  object.

  Fields:
    name: Required. Name of the shared flow revision to deploy in the
      following format: `organizations/{org}/environments/{env}/sharedflows/{s
      haredflow}/revisions/{rev}`
    override: Flag that specifies whether the new deployment replaces other
      deployed revisions of the shared flow in the environment. Set `override`
      to `true` to replace other deployed revisions. By default, `override` is
      `false` and the deployment is rejected if other revisions of the shared
      flow are deployed in the environment.
    serviceAccount: Google Cloud IAM service account. The service account
      represents the identity of the deployed proxy, and determines what
      permissions it has. The format must be
      `{ACCOUNT_ID}@{PROJECT}.iam.gserviceaccount.com`.
  """
    name = _messages.StringField(1, required=True)
    override = _messages.BooleanField(2)
    serviceAccount = _messages.StringField(3)