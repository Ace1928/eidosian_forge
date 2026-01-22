from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsAuthorizedCertificatesCreateRequest(_messages.Message):
    """A AppengineAppsAuthorizedCertificatesCreateRequest object.

  Fields:
    authorizedCertificate: A AuthorizedCertificate resource to be passed as
      the request body.
    parent: Name of the parent Application resource. Example: apps/myapp.
  """
    authorizedCertificate = _messages.MessageField('AuthorizedCertificate', 1)
    parent = _messages.StringField(2, required=True)