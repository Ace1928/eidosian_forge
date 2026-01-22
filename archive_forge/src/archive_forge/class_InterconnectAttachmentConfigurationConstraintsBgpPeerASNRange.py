from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectAttachmentConfigurationConstraintsBgpPeerASNRange(_messages.Message):
    """A InterconnectAttachmentConfigurationConstraintsBgpPeerASNRange object.

  Fields:
    max: A integer attribute.
    min: A integer attribute.
  """
    max = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    min = _messages.IntegerField(2, variant=_messages.Variant.UINT32)