from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsApisDeploymentsListRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsApisDeploymentsListRequest object.

  Fields:
    parent: Required. Name representing an API proxy in an environment in the
      following format: `organizations/{org}/environments/{env}/apis/{api}`
  """
    parent = _messages.StringField(1, required=True)