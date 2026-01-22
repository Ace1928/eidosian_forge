from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendMetastore(_messages.Message):
    """Represents a backend metastore for the federation.

  Enums:
    MetastoreTypeValueValuesEnum: The type of the backend metastore.

  Fields:
    metastoreType: The type of the backend metastore.
    name: The relative resource name of the metastore that is being federated.
      The formats of the relative resource names for the currently supported
      metastores are listed below: BigQuery projects/{project_id} Dataproc
      Metastore
      projects/{project_id}/locations/{location}/services/{service_id}
  """

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
    metastoreType = _messages.EnumField('MetastoreTypeValueValuesEnum', 1)
    name = _messages.StringField(2)