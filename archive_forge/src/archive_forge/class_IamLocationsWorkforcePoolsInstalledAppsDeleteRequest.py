from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsInstalledAppsDeleteRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsInstalledAppsDeleteRequest object.

  Fields:
    name: Required. The name of the workforce pool installed app to delete.
      Format: `locations/{location}/workforcePools/{workforce_pool}/installedA
      pps/{installed_app}`
    validateOnly: Optional. If set, validate the request and preview the
      response, but do not actually post it.
  """
    name = _messages.StringField(1, required=True)
    validateOnly = _messages.BooleanField(2)