from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrafeasV1beta1IntotoArtifact(_messages.Message):
    """A GrafeasV1beta1IntotoArtifact object.

  Fields:
    hashes: A ArtifactHashes attribute.
    resourceUri: A string attribute.
  """
    hashes = _messages.MessageField('ArtifactHashes', 1)
    resourceUri = _messages.StringField(2)