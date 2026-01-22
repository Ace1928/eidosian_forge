from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeInsight(_messages.Message):
    """This insight is a recommendation to upgrade a given library to the
  specified version, in order to avoid dependencies on non-SDK APIs.

  Fields:
    packageName: The name of the package to be upgraded.
    upgradeToVersion: The suggested version to upgrade to. Optional: In case
      we are not sure which version solves this problem
  """
    packageName = _messages.StringField(1)
    upgradeToVersion = _messages.StringField(2)