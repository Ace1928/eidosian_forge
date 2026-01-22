from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsArchiveDeploymentsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsArchiveDeploymentsCreateRequest object.

  Fields:
    googleCloudApigeeV1ArchiveDeployment: A
      GoogleCloudApigeeV1ArchiveDeployment resource to be passed as the
      request body.
    parent: Required. The Environment this Archive Deployment will be created
      in.
  """
    googleCloudApigeeV1ArchiveDeployment = _messages.MessageField('GoogleCloudApigeeV1ArchiveDeployment', 1)
    parent = _messages.StringField(2, required=True)