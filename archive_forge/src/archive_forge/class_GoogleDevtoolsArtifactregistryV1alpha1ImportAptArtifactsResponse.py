from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1alpha1ImportAptArtifactsResponse(_messages.Message):
    """The response message from importing APT artifacts.

  Fields:
    aptArtifacts: The Apt artifacts imported.
    errors: Detailed error info for packages that were not imported.
  """
    aptArtifacts = _messages.MessageField('GoogleDevtoolsArtifactregistryV1alpha1AptArtifact', 1, repeated=True)
    errors = _messages.MessageField('GoogleDevtoolsArtifactregistryV1alpha1ImportAptArtifactsErrorInfo', 2, repeated=True)