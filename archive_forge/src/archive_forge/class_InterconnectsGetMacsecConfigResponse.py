from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectsGetMacsecConfigResponse(_messages.Message):
    """Response for the InterconnectsGetMacsecConfigRequest.

  Fields:
    etag: end_interface: MixerGetResponseWithEtagBuilder
    result: A InterconnectMacsecConfig attribute.
  """
    etag = _messages.StringField(1)
    result = _messages.MessageField('InterconnectMacsecConfig', 2)