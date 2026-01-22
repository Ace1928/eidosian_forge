from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageConfig(_messages.Message):
    """ImageConfig defines the control plane images to run.

  Fields:
    stableImage: The stable image that the remote agent will fallback to if
      the target image fails.
    targetImage: The initial image the remote agent will attempt to run for
      the control plane.
  """
    stableImage = _messages.StringField(1)
    targetImage = _messages.StringField(2)