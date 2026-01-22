from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SurvivabilityConfig(_messages.Message):
    """Configuration of the cluster survivability, e.g., for the case when
  network connectivity is lost.

  Fields:
    offlineRebootTtl: Optional. Time period that allows the cluster nodes to
      be rebooted and become functional without network connectivity to
      Google. The default 0 means not allowed. The maximum is 7 days.
  """
    offlineRebootTtl = _messages.StringField(1)