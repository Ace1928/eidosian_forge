from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildArtifact(_messages.Message):
    """Description of an a image to use during Skaffold rendering.

  Fields:
    image: Image name in Skaffold configuration.
    tag: Image tag to use. This will generally be the full path to an image,
      such as "gcr.io/my-project/busybox:1.2.3" or "gcr.io/my-
      project/busybox@sha256:abc123".
  """
    image = _messages.StringField(1)
    tag = _messages.StringField(2)