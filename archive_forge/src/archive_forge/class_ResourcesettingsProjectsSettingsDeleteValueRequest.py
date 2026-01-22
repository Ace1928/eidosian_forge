from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResourcesettingsProjectsSettingsDeleteValueRequest(_messages.Message):
    """A ResourcesettingsProjectsSettingsDeleteValueRequest object.

  Fields:
    ignoreReadOnly: A flag that allows the deletion of the value of a
      `read_only` setting. WARNING: use at your own risk. Deleting the value
      of a read only setting is an irreversible action (i.e., it cannot be
      created again).
    name: The name of the setting value to delete. See SettingValue for naming
      requirements.
  """
    ignoreReadOnly = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)