from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsAppgroupsGetRequest(_messages.Message):
    """A ApigeeOrganizationsAppgroupsGetRequest object.

  Fields:
    name: Required. Name of the AppGroup. Use the following structure in your
      request: `organizations/{org}/appgroups/{app_group_name}`
  """
    name = _messages.StringField(1, required=True)