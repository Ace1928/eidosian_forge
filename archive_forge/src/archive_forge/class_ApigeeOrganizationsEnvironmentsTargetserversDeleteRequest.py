from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsTargetserversDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsTargetserversDeleteRequest object.

  Fields:
    name: Required. The name of the TargetServer to delete. Must be of the
      form `organizations/{org}/environments/{env}/targetservers/{target_serve
      r_id}`.
  """
    name = _messages.StringField(1, required=True)