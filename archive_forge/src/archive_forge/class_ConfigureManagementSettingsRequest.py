from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigureManagementSettingsRequest(_messages.Message):
    """Request for the `ConfigureManagementSettings` method.

  Fields:
    managementSettings: Fields of the `ManagementSettings` to update.
    updateMask: Required. The field mask describing which fields to update as
      a comma-separated list. For example, if only the transfer lock is being
      updated, the `update_mask` is `"transfer_lock_state"`.
  """
    managementSettings = _messages.MessageField('ManagementSettings', 1)
    updateMask = _messages.StringField(2)