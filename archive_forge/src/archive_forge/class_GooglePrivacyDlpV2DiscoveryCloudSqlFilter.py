from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DiscoveryCloudSqlFilter(_messages.Message):
    """Determines what tables will have profiles generated within an
  organization or project. Includes the ability to filter by regular
  expression patterns on project ID, location, instance, database, and
  database resource name.

  Fields:
    collection: A specific set of database resources for this filter to apply
      to.
    databaseResourceReference: The database resource to scan. Targets
      including this can only include one target (the target with this
      database resource reference).
    others: Catch-all. This should always be the last target in the list
      because anything above it will apply first. Should only appear once in a
      configuration. If none is specified, a default one will be added
      automatically.
  """
    collection = _messages.MessageField('GooglePrivacyDlpV2DatabaseResourceCollection', 1)
    databaseResourceReference = _messages.MessageField('GooglePrivacyDlpV2DatabaseResourceReference', 2)
    others = _messages.MessageField('GooglePrivacyDlpV2AllOtherDatabaseResources', 3)