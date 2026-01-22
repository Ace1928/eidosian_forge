from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1alpha1UploadYumArtifactResponse(_messages.Message):
    """The response of the completed artifact upload operation. This response
  is contained in the Operation and available to users.

  Fields:
    yumArtifacts: The Yum artifacts updated.
  """
    yumArtifacts = _messages.MessageField('GoogleDevtoolsArtifactregistryV1alpha1YumArtifact', 1, repeated=True)