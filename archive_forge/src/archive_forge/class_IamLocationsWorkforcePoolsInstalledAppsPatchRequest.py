from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsInstalledAppsPatchRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsInstalledAppsPatchRequest object.

  Fields:
    name: Immutable. The resource name of the workforce pool installed app.
      Format: `locations/{location}/workforcePools/{workforce_pool}/installedA
      pps/{installed_app}`
    updateMask: Required. The list of fields to update.
    workforcePoolInstalledApp: A WorkforcePoolInstalledApp resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    workforcePoolInstalledApp = _messages.MessageField('WorkforcePoolInstalledApp', 3)