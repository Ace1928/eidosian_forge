from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityReportsGetResultRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSecurityReportsGetResultRequest object.

  Fields:
    name: Required. Name of the security report result to get. Must be of the
      form `organizations/{org}/environments/{env}/securityReports/{reportId}/
      result`.
  """
    name = _messages.StringField(1, required=True)