from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudfunctionsProjectsLocationsFunctionsGenerateDownloadUrlRequest(_messages.Message):
    """A CloudfunctionsProjectsLocationsFunctionsGenerateDownloadUrlRequest
  object.

  Fields:
    generateDownloadUrlRequest: A GenerateDownloadUrlRequest resource to be
      passed as the request body.
    name: Required. The name of function for which source code Google Cloud
      Storage signed URL should be generated.
  """
    generateDownloadUrlRequest = _messages.MessageField('GenerateDownloadUrlRequest', 1)
    name = _messages.StringField(2, required=True)