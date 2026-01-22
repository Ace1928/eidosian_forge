from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DeployedIndexAuthConfig(_messages.Message):
    """Used to set up the auth on the DeployedIndex's private endpoint.

  Fields:
    authProvider: Defines the authentication provider that the DeployedIndex
      uses.
  """
    authProvider = _messages.MessageField('GoogleCloudAiplatformV1DeployedIndexAuthConfigAuthProvider', 1)