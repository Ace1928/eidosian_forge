from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsArchiveDeploymentsGetRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsArchiveDeploymentsGetRequest object.

  Fields:
    name: Required. Name of the Archive Deployment in the following format:
      `organizations/{org}/environments/{env}/archiveDeployments/{id}`.
  """
    name = _messages.StringField(1, required=True)