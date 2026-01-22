from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlActiveDirectoryConfig(_messages.Message):
    """Active Directory configuration, relevant only for Cloud SQL for SQL
  Server.

  Fields:
    domain: The name of the domain (e.g., mydomain.com).
    kind: This is always sql#activeDirectoryConfig.
  """
    domain = _messages.StringField(1)
    kind = _messages.StringField(2)