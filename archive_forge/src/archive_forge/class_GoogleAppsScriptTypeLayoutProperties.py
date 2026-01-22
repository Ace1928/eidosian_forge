from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsScriptTypeLayoutProperties(_messages.Message):
    """Card layout properties shared across all add-on host applications.

  Fields:
    primaryColor: The primary color of the add-on. It sets the color of the
      toolbar. If no primary color is set, the default value provided by the
      framework is used.
    secondaryColor: The secondary color of the add-on. It sets the color of
      buttons. If the primary color is set but no secondary color is set, the
      secondary color is the same as the primary color. If neither primary
      color nor secondary color is set, the default value provided by the
      framework is used.
    useNewMaterialDesign: Enables material design for cards.
  """
    primaryColor = _messages.StringField(1)
    secondaryColor = _messages.StringField(2)
    useNewMaterialDesign = _messages.BooleanField(3)