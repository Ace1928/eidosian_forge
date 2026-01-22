from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SidecarState(_messages.Message):
    """The state of a sidecar.

  Fields:
    containerName: Name of the container.
    imageId: ID of the image.
    name: Name of the Sidecar.
    running: Details about a running container.
    terminated: Details about a terminated container.
    waiting: Details about a waiting container.
  """
    containerName = _messages.StringField(1)
    imageId = _messages.StringField(2)
    name = _messages.StringField(3)
    running = _messages.MessageField('ContainerStateRunning', 4)
    terminated = _messages.MessageField('ContainerStateTerminated', 5)
    waiting = _messages.MessageField('ContainerStateWaiting', 6)