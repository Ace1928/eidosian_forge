from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookupServiceProjectAttachmentResponse(_messages.Message):
    """Response for LookupServiceProjectAttachment.

  Fields:
    serviceProjectAttachment: Service project attachment for a project if
      exists, empty otherwise.
  """
    serviceProjectAttachment = _messages.MessageField('ServiceProjectAttachment', 1)