from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModerateTextRequest(_messages.Message):
    """The document moderation request message.

  Fields:
    document: Required. Input document.
  """
    document = _messages.MessageField('Document', 1)