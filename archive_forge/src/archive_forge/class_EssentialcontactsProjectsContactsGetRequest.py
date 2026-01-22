from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EssentialcontactsProjectsContactsGetRequest(_messages.Message):
    """A EssentialcontactsProjectsContactsGetRequest object.

  Fields:
    name: Required. The name of the contact to retrieve. Format:
      organizations/{organization_id}/contacts/{contact_id},
      folders/{folder_id}/contacts/{contact_id} or
      projects/{project_id}/contacts/{contact_id}
  """
    name = _messages.StringField(1, required=True)