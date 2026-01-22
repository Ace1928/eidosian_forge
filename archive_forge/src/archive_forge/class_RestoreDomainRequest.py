from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestoreDomainRequest(_messages.Message):
    """RestoreDomainRequest is the request received by RestoreDomain rpc

  Fields:
    backupId: Required. ID of the backup to be restored
  """
    backupId = _messages.StringField(1)