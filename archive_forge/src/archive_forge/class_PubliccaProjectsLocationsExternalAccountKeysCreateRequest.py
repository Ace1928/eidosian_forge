from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubliccaProjectsLocationsExternalAccountKeysCreateRequest(_messages.Message):
    """A PubliccaProjectsLocationsExternalAccountKeysCreateRequest object.

  Fields:
    externalAccountKey: A ExternalAccountKey resource to be passed as the
      request body.
    parent: Required. The parent resource where this external_account_key will
      be created. Format: projects/[project_id]/locations/[location]. At
      present only the "global" location is supported.
  """
    externalAccountKey = _messages.MessageField('ExternalAccountKey', 1)
    parent = _messages.StringField(2, required=True)