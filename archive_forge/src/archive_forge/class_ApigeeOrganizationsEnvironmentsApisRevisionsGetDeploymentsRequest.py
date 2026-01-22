from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsApisRevisionsGetDeploymentsRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsApisRevisionsGetDeploymentsRequest
  object.

  Fields:
    name: Required. Name representing an API proxy revision in an environment
      in the following format:
      `organizations/{org}/environments/{env}/apis/{api}/revisions/{rev}`
  """
    name = _messages.StringField(1, required=True)