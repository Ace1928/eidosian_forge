from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageAllowlist(_messages.Message):
    """Images that are exempted from normal checks based on name pattern only.

  Fields:
    allowPattern: Required. A disjunction of image patterns to allow. If any
      of these patterns match, then the image is considered exempted by this
      allowlist.
  """
    allowPattern = _messages.StringField(1, repeated=True)