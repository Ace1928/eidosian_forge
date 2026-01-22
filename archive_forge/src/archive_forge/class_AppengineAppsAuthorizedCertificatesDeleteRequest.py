from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsAuthorizedCertificatesDeleteRequest(_messages.Message):
    """A AppengineAppsAuthorizedCertificatesDeleteRequest object.

  Fields:
    name: Name of the resource to delete. Example:
      apps/myapp/authorizedCertificates/12345.
  """
    name = _messages.StringField(1, required=True)