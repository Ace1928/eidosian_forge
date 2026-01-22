from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsDeploymentsGetRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsDeploymentsGetRequest object.

  Fields:
    name: Required. Name of the api proxy or the shared flow deployment. Use
      the following structure in your request:
      `organizations/{org}/environments/{env}/deployments/{deployment}`
  """
    name = _messages.StringField(1, required=True)