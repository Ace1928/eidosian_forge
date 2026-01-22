from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityActionsGetRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSecurityActionsGetRequest object.

  Fields:
    name: Required. The fully qualified name of the SecurityAction to
      retrieve. Format:
      organizations/{org}/environments/{env}/securityActions/{security_action}
  """
    name = _messages.StringField(1, required=True)