from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IsInstanceUpgradeableResponse(_messages.Message):
    """Response for checking if a notebook instance is upgradeable.

  Fields:
    upgradeImage: The new image self link this instance will be upgraded to if
      calling the upgrade endpoint. This field will only be populated if field
      upgradeable is true.
    upgradeInfo: Additional information about upgrade.
    upgradeVersion: The version this instance will be upgraded to if calling
      the upgrade endpoint. This field will only be populated if field
      upgradeable is true.
    upgradeable: If an instance is upgradeable.
  """
    upgradeImage = _messages.StringField(1)
    upgradeInfo = _messages.StringField(2)
    upgradeVersion = _messages.StringField(3)
    upgradeable = _messages.BooleanField(4)