from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceSelector(_messages.Message):
    """Specifies the resource to analyze for access policies, which may be set
  directly on the resource, or on ancestors such as organizations, folders or
  projects.

  Fields:
    fullResourceName: Required. The [full resource name]
      (https://cloud.google.com/asset-inventory/docs/resource-name-format) of
      a resource of [supported resource types](https://cloud.google.com/asset-
      inventory/docs/supported-asset-types#analyzable_asset_types).
  """
    fullResourceName = _messages.StringField(1)