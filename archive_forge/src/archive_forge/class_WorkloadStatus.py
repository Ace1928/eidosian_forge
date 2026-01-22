from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadStatus(_messages.Message):
    """Workload status.

  Fields:
    siteVersion: Output only. SiteVersion running in the workload cluster.
    status: Output only. Status.
  """
    siteVersion = _messages.MessageField('SiteVersion', 1)
    status = _messages.StringField(2)