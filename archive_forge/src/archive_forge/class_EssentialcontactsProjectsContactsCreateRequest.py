from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EssentialcontactsProjectsContactsCreateRequest(_messages.Message):
    """A EssentialcontactsProjectsContactsCreateRequest object.

  Fields:
    googleCloudEssentialcontactsV1Contact: A
      GoogleCloudEssentialcontactsV1Contact resource to be passed as the
      request body.
    parent: Required. The resource to save this contact for. Format:
      organizations/{organization_id}, folders/{folder_id} or
      projects/{project_id}
  """
    googleCloudEssentialcontactsV1Contact = _messages.MessageField('GoogleCloudEssentialcontactsV1Contact', 1)
    parent = _messages.StringField(2, required=True)