from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HydrationStatus(_messages.Message):
    """Hydration status.

  Fields:
    siteVersion: Output only. SiteVersion Hydration is targeting.
    status: Output only. Status.
  """
    siteVersion = _messages.MessageField('SiteVersion', 1)
    status = _messages.StringField(2)