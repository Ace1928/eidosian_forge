from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1DataplexExternalTable(_messages.Message):
    """External table registered by Dataplex. Dataplex publishes data
  discovered from an asset into multiple other systems (BigQuery, DPMS) in
  form of tables. We call them "external tables". External tables are also
  synced into the Data Catalog. This message contains pointers to those
  external tables (fully qualified name, resource name et cetera) within the
  Data Catalog.

  Enums:
    SystemValueValuesEnum: Service in which the external table is registered.

  Fields:
    dataCatalogEntry: Name of the Data Catalog entry representing the external
      table.
    fullyQualifiedName: Fully qualified name (FQN) of the external table.
    googleCloudResource: Google Cloud resource name of the external table.
    system: Service in which the external table is registered.
  """

    class SystemValueValuesEnum(_messages.Enum):
        """Service in which the external table is registered.

    Values:
      INTEGRATED_SYSTEM_UNSPECIFIED: Default unknown system.
      BIGQUERY: BigQuery.
      CLOUD_PUBSUB: Cloud Pub/Sub.
      DATAPROC_METASTORE: Dataproc Metastore.
      DATAPLEX: Dataplex.
      CLOUD_SPANNER: Cloud Spanner
      CLOUD_BIGTABLE: Cloud Bigtable
      CLOUD_SQL: Cloud Sql
      LOOKER: Looker
      VERTEX_AI: Vertex AI
    """
        INTEGRATED_SYSTEM_UNSPECIFIED = 0
        BIGQUERY = 1
        CLOUD_PUBSUB = 2
        DATAPROC_METASTORE = 3
        DATAPLEX = 4
        CLOUD_SPANNER = 5
        CLOUD_BIGTABLE = 6
        CLOUD_SQL = 7
        LOOKER = 8
        VERTEX_AI = 9
    dataCatalogEntry = _messages.StringField(1)
    fullyQualifiedName = _messages.StringField(2)
    googleCloudResource = _messages.StringField(3)
    system = _messages.EnumField('SystemValueValuesEnum', 4)