from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoUpgradeOptions(_messages.Message):
    """AutoUpgradeOptions defines the set of options for the user to control
  how the Auto Upgrades will proceed.

  Fields:
    autoUpgradeStartTime: [Output only] This field is set when upgrades are
      about to commence with the approximate start time for the upgrades, in
      [RFC3339](https://www.ietf.org/rfc/rfc3339.txt) text format.
    description: [Output only] This field is set when upgrades are about to
      commence with the description of the upgrade.
  """
    autoUpgradeStartTime = _messages.StringField(1)
    description = _messages.StringField(2)