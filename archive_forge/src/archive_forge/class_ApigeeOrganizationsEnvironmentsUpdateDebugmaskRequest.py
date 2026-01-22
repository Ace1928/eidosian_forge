from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsUpdateDebugmaskRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsUpdateDebugmaskRequest object.

  Fields:
    googleCloudApigeeV1DebugMask: A GoogleCloudApigeeV1DebugMask resource to
      be passed as the request body.
    name: Name of the debug mask.
    replaceRepeatedFields: Boolean flag that specifies whether to replace
      existing values in the debug mask when doing an update. Set to true to
      replace existing values. The default behavior is to append the values
      (false).
    updateMask: Field debug mask to support partial updates.
  """
    googleCloudApigeeV1DebugMask = _messages.MessageField('GoogleCloudApigeeV1DebugMask', 1)
    name = _messages.StringField(2, required=True)
    replaceRepeatedFields = _messages.BooleanField(3)
    updateMask = _messages.StringField(4)