from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RewriteResponse(_messages.Message):
    """A rewrite response.

  Fields:
    done: true if the copy is finished; otherwise, false if the copy is in
      progress. This property is always present in the response.
    kind: The kind of item this is.
    objectSize: The total size of the object being copied in bytes. This
      property is always present in the response.
    resource: A resource containing the metadata for the copied-to object.
      This property is present in the response only when copying completes.
    rewriteToken: A token to use in subsequent requests to continue copying
      data. This token is present in the response only when there is more data
      to copy.
    totalBytesRewritten: The total bytes written so far, which can be used to
      provide a waiting user with a progress indicator. This property is
      always present in the response.
  """
    done = _messages.BooleanField(1)
    kind = _messages.StringField(2, default=u'storage#rewriteResponse')
    objectSize = _messages.IntegerField(3)
    resource = _messages.MessageField('Object', 4)
    rewriteToken = _messages.StringField(5)
    totalBytesRewritten = _messages.IntegerField(6)