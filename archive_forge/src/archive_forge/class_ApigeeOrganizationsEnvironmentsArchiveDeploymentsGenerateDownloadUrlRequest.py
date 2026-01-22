from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsArchiveDeploymentsGenerateDownloadUrlRequest(_messages.Message):
    """A
  ApigeeOrganizationsEnvironmentsArchiveDeploymentsGenerateDownloadUrlRequest
  object.

  Fields:
    googleCloudApigeeV1GenerateDownloadUrlRequest: A
      GoogleCloudApigeeV1GenerateDownloadUrlRequest resource to be passed as
      the request body.
    name: Required. The name of the Archive Deployment you want to download.
  """
    googleCloudApigeeV1GenerateDownloadUrlRequest = _messages.MessageField('GoogleCloudApigeeV1GenerateDownloadUrlRequest', 1)
    name = _messages.StringField(2, required=True)