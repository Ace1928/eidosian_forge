from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ViolatingResource(_messages.Message):
    """Information regarding any resource causing the vulnerability such as
  JavaScript sources, image, audio files, etc.

  Fields:
    contentType: The MIME type of this resource.
    resourceUrl: URL of this violating resource.
  """
    contentType = _messages.StringField(1)
    resourceUrl = _messages.StringField(2)