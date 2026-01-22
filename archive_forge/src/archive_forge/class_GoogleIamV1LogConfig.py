from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV1LogConfig(_messages.Message):
    """Specifies what kind of log the caller must write

  Fields:
    cloudAudit: Cloud audit options.
    counter: Counter options.
    dataAccess: Data access options.
  """
    cloudAudit = _messages.MessageField('GoogleIamV1LogConfigCloudAuditOptions', 1)
    counter = _messages.MessageField('GoogleIamV1LogConfigCounterOptions', 2)
    dataAccess = _messages.MessageField('GoogleIamV1LogConfigDataAccessOptions', 3)