from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChannelConnection(_messages.Message):
    """A representation of the ChannelConnection resource. A ChannelConnection
  is a resource which event providers create during the activation process to
  establish a connection between the provider and the subscriber channel.

  Fields:
    activationToken: Input only. Activation token for the channel. The token
      will be used during the creation of ChannelConnection to bind the
      channel with the provider project. This field will not be stored in the
      provider resource.
    channel: Required. The name of the connected subscriber Channel. This is a
      weak reference to avoid cross project and cross accounts references.
      This must be in
      `projects/{project}/location/{location}/channels/{channel_id}` format.
    createTime: Output only. The creation time.
    name: Required. The name of the connection.
    uid: Output only. Server assigned ID of the resource. The server
      guarantees uniqueness and immutability until deleted.
    updateTime: Output only. The last-modified time.
  """
    activationToken = _messages.StringField(1)
    channel = _messages.StringField(2)
    createTime = _messages.StringField(3)
    name = _messages.StringField(4)
    uid = _messages.StringField(5)
    updateTime = _messages.StringField(6)