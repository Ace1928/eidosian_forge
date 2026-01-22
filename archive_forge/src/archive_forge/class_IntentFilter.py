from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IntentFilter(_messages.Message):
    """The section of an tag.
  https://developer.android.com/guide/topics/manifest/intent-filter-
  element.html

  Fields:
    actionNames: The android:name value of the tag.
    categoryNames: The android:name value of the tag.
    mimeType: The android:mimeType value of the tag.
  """
    actionNames = _messages.StringField(1, repeated=True)
    categoryNames = _messages.StringField(2, repeated=True)
    mimeType = _messages.StringField(3)