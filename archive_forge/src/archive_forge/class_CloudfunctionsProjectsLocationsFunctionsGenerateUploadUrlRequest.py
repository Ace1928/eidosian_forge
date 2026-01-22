from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudfunctionsProjectsLocationsFunctionsGenerateUploadUrlRequest(_messages.Message):
    """A CloudfunctionsProjectsLocationsFunctionsGenerateUploadUrlRequest
  object.

  Fields:
    generateUploadUrlRequest: A GenerateUploadUrlRequest resource to be passed
      as the request body.
    parent: Required. The project and location in which the Google Cloud
      Storage signed URL should be generated, specified in the format
      `projects/*/locations/*`.
  """
    generateUploadUrlRequest = _messages.MessageField('GenerateUploadUrlRequest', 1)
    parent = _messages.StringField(2, required=True)