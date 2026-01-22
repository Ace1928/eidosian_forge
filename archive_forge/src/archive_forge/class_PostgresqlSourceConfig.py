from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PostgresqlSourceConfig(_messages.Message):
    """PostgreSQL data source configuration

  Fields:
    excludeObjects: PostgreSQL objects to exclude from the stream.
    includeObjects: PostgreSQL objects to include in the stream.
    maxConcurrentBackfillTasks: Maximum number of concurrent backfill tasks.
      The number should be non negative. If not set (or set to 0), the
      system's default value will be used.
    publication: Required. The name of the publication that includes the set
      of all tables that are defined in the stream's include_objects.
    replicationSlot: Required. Immutable. The name of the logical replication
      slot that's configured with the pgoutput plugin.
  """
    excludeObjects = _messages.MessageField('PostgresqlRdbms', 1)
    includeObjects = _messages.MessageField('PostgresqlRdbms', 2)
    maxConcurrentBackfillTasks = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    publication = _messages.StringField(4)
    replicationSlot = _messages.StringField(5)