from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreTypeValueValuesEnum(_messages.Enum):
    """The type of the backend metastore.

    Values:
      METASTORE_TYPE_UNSPECIFIED: The metastore type is not set.
      DATAPLEX: The backend metastore is Dataplex.
      BIGQUERY: The backend metastore is BigQuery.
      DATAPROC_METASTORE: The backend metastore is Dataproc Metastore.
    """
    METASTORE_TYPE_UNSPECIFIED = 0
    DATAPLEX = 1
    BIGQUERY = 2
    DATAPROC_METASTORE = 3