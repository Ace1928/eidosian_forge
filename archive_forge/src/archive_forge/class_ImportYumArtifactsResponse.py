from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportYumArtifactsResponse(_messages.Message):
    """The response message from importing YUM artifacts.

  Fields:
    errors: Detailed error info for packages that were not imported.
    yumArtifacts: The yum artifacts imported.
  """
    errors = _messages.MessageField('ImportYumArtifactsErrorInfo', 1, repeated=True)
    yumArtifacts = _messages.MessageField('YumArtifact', 2, repeated=True)