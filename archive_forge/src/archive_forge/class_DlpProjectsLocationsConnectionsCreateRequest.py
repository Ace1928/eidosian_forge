from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsConnectionsCreateRequest(_messages.Message):
    """A DlpProjectsLocationsConnectionsCreateRequest object.

  Fields:
    googlePrivacyDlpV2CreateConnectionRequest: A
      GooglePrivacyDlpV2CreateConnectionRequest resource to be passed as the
      request body.
    parent: Required. Parent resource name in the format:
      "projects/{project}/locations/{location}".
  """
    googlePrivacyDlpV2CreateConnectionRequest = _messages.MessageField('GooglePrivacyDlpV2CreateConnectionRequest', 1)
    parent = _messages.StringField(2, required=True)