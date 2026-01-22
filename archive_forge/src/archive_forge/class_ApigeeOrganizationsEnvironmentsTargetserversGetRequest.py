from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsTargetserversGetRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsTargetserversGetRequest object.

  Fields:
    name: Required. The name of the TargetServer to get. Must be of the form
      `organizations/{org}/environments/{env}/targetservers/{target_server_id}
      `.
  """
    name = _messages.StringField(1, required=True)