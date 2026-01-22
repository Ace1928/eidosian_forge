from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EssentialcontactsOrganizationsContactsVerifyRequest(_messages.Message):
    """A EssentialcontactsOrganizationsContactsVerifyRequest object.

  Fields:
    googleCloudEssentialcontactsV1alpha1VerifyContactRequest: A
      GoogleCloudEssentialcontactsV1alpha1VerifyContactRequest resource to be
      passed as the request body.
    name: Required. The name of the contact to verify. Format:
      organizations/{organization_id}/contacts/{contact_id},
      folders/{folder_id}/contacts/{contact_id} or
      projects/{project_id}/contacts/{contact_id}
  """
    googleCloudEssentialcontactsV1alpha1VerifyContactRequest = _messages.MessageField('GoogleCloudEssentialcontactsV1alpha1VerifyContactRequest', 1)
    name = _messages.StringField(2, required=True)