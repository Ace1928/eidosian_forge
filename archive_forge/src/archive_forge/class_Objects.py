from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Objects(_messages.Message):
    """A list of objects.

  Fields:
    items: The list of items.
    kind: The kind of item this is. For lists of objects, this is always
      storage#objects.
    nextPageToken: The continuation token, used to page through large result
      sets. Provide this value in a subsequent request to return the next page
      of results.
    prefixes: The list of prefixes of objects matching-but-not-listed up to
      and including the requested delimiter.
  """
    items = _messages.MessageField('Object', 1, repeated=True)
    kind = _messages.StringField(2, default=u'storage#objects')
    nextPageToken = _messages.StringField(3)
    prefixes = _messages.StringField(4, repeated=True)