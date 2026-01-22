from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpAppconnectorsV1alphaImageConfig(_messages.Message):
    """ImageConfig defines the control plane images to run.

  Fields:
    stableImage: The stable image that the remote agent will fallback to if
      the target image fails. Format would be a gcr image path, e.g.:
      gcr.io/PROJECT-ID/my-image:tag1
    targetImage: The initial image the remote agent will attempt to run for
      the control plane. Format would be a gcr image path, e.g.:
      gcr.io/PROJECT-ID/my-image:tag1
  """
    stableImage = _messages.StringField(1)
    targetImage = _messages.StringField(2)