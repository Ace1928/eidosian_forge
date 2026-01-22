from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesRulesPatchRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesRulesPatchRequest object.

  Fields:
    name: A user-specified resource name of the rule
      `projects/{project}/locations/{location}/assetTypes/{type}/rules/{rule}`
      . Here {rule} is a resource id. Detailed rules for a resource id are: 1.
      1 character minimum, 63 characters maximum 2. only contains letters,
      digits, underscore and hyphen 3. starts with a letter if length == 1,
      starts with a letter or underscore if length > 1
    rule: A Rule resource to be passed as the request body.
    updateMask: Required. Comma-separated list of fields to be updated.
  """
    name = _messages.StringField(1, required=True)
    rule = _messages.MessageField('Rule', 2)
    updateMask = _messages.StringField(3)