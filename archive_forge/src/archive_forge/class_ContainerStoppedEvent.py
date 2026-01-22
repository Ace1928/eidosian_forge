from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerStoppedEvent(_messages.Message):
    """An event generated when a container exits.

  Fields:
    actionId: The numeric ID of the action that started this container.
    exitStatus: The exit status of the container.
    stderr: The tail end of any content written to standard error by the
      container. If the content emits large amounts of debugging noise or
      contains sensitive information, you can prevent the content from being
      printed by setting the `DISABLE_STANDARD_ERROR_CAPTURE` flag. Note that
      only a small amount of the end of the stream is captured here. The
      entire stream is stored in the `/google/logs` directory mounted into
      each action, and can be copied off the machine as described elsewhere.
  """
    actionId = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    exitStatus = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    stderr = _messages.StringField(3)