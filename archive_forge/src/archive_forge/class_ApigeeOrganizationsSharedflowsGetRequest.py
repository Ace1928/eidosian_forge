from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSharedflowsGetRequest(_messages.Message):
    """A ApigeeOrganizationsSharedflowsGetRequest object.

  Fields:
    name: Required. The name of the shared flow to get. Must be of the form:
      `organizations/{organization_id}/sharedflows/{shared_flow_id}`
  """
    name = _messages.StringField(1, required=True)