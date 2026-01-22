from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsAppgroupsAppsKeysApiproductsDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsAppgroupsAppsKeysApiproductsDeleteRequest object.

  Fields:
    name: Required. Parent of the AppGroup app key. Use the following
      structure in your request: `organizations/{org}/appgroups/{app_group_nam
      e}/apps/{app}/keys/{key}/apiproducts/{apiproduct}`
  """
    name = _messages.StringField(1, required=True)