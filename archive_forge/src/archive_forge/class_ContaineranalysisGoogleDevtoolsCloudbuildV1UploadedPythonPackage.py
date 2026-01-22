from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisGoogleDevtoolsCloudbuildV1UploadedPythonPackage(_messages.Message):
    """Artifact uploaded using the PythonPackage directive.

  Fields:
    fileHashes: Hash types and values of the Python Artifact.
    pushTiming: Output only. Stores timing information for pushing the
      specified artifact.
    uri: URI of the uploaded artifact.
  """
    fileHashes = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1FileHashes', 1)
    pushTiming = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1TimeSpan', 2)
    uri = _messages.StringField(3)