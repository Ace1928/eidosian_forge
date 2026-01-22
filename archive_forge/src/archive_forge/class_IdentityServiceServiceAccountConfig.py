from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityServiceServiceAccountConfig(_messages.Message):
    """Contains the credentials of the service account which is authorized to
  perform the LDAP search in the directory. The credentials can be supplied by
  the combination of the DN and password or the client certificate.

  Fields:
    simpleBindCredentials: Credentials for basic auth.
  """
    simpleBindCredentials = _messages.MessageField('IdentityServiceSimpleBindCredentials', 1)