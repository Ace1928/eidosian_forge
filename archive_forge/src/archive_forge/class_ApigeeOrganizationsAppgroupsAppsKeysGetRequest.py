from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsAppgroupsAppsKeysGetRequest(_messages.Message):
    """A ApigeeOrganizationsAppgroupsAppsKeysGetRequest object.

  Fields:
    name: Required. Name of the AppGroup app key. Use the following structure
      in your request:
      `organizations/{org}/appgroups/{app_group_name}/apps/{app}/keys/{key}`
  """
    name = _messages.StringField(1, required=True)