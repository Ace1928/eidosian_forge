from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkforceIdentityBasedManagementURI(_messages.Message):
    """ManagementURI depending on the Workforce Identity i.e. either 1p or 3p.

  Fields:
    firstPartyManagementUri: Output only. First party Management URI for
      Google Identities.
    thirdPartyManagementUri: Output only. Third party Management URI for
      External Identity Providers.
  """
    firstPartyManagementUri = _messages.StringField(1)
    thirdPartyManagementUri = _messages.StringField(2)