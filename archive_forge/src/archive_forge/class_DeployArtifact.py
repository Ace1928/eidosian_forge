from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeployArtifact(_messages.Message):
    """The artifacts produced by a deploy operation.

  Fields:
    artifactUri: Output only. URI of a directory containing the artifacts. All
      paths are relative to this location.
    manifestPaths: Output only. File paths of the manifests applied during the
      deploy operation relative to the URI.
  """
    artifactUri = _messages.StringField(1)
    manifestPaths = _messages.StringField(2, repeated=True)