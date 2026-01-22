from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnableMigrationRequest(_messages.Message):
    """EnableMigrationRequest is the request message for EnableMigration
  method.

  Fields:
    migratingDomains: Required. List of the on-prem domains to be migrated.
  """
    migratingDomains = _messages.MessageField('OnPremDomainDetails', 1, repeated=True)