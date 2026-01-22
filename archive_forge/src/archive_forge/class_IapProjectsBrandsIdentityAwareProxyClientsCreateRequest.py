from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IapProjectsBrandsIdentityAwareProxyClientsCreateRequest(_messages.Message):
    """A IapProjectsBrandsIdentityAwareProxyClientsCreateRequest object.

  Fields:
    identityAwareProxyClient: A IdentityAwareProxyClient resource to be passed
      as the request body.
    parent: Required. Path to create the client in. In the following format:
      projects/{project_number/id}/brands/{brand}. The project must belong to
      a G Suite account.
  """
    identityAwareProxyClient = _messages.MessageField('IdentityAwareProxyClient', 1)
    parent = _messages.StringField(2, required=True)