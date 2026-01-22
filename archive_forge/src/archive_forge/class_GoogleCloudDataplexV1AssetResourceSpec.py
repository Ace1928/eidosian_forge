from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1AssetResourceSpec(_messages.Message):
    """Identifies the cloud resource that is referenced by this asset.

  Enums:
    ReadAccessModeValueValuesEnum: Optional. Determines how read permissions
      are handled for each asset and their associated tables. Only available
      to storage buckets assets.
    TypeValueValuesEnum: Required. Immutable. Type of resource.

  Fields:
    name: Immutable. Relative name of the cloud resource that contains the
      data that is being managed within a lake. For example:
      projects/{project_number}/buckets/{bucket_id}
      projects/{project_number}/datasets/{dataset_id}
    readAccessMode: Optional. Determines how read permissions are handled for
      each asset and their associated tables. Only available to storage
      buckets assets.
    type: Required. Immutable. Type of resource.
  """

    class ReadAccessModeValueValuesEnum(_messages.Enum):
        """Optional. Determines how read permissions are handled for each asset
    and their associated tables. Only available to storage buckets assets.

    Values:
      ACCESS_MODE_UNSPECIFIED: Access mode unspecified.
      DIRECT: Default. Data is accessed directly using storage APIs.
      MANAGED: Data is accessed through a managed interface using BigQuery
        APIs.
    """
        ACCESS_MODE_UNSPECIFIED = 0
        DIRECT = 1
        MANAGED = 2

    class TypeValueValuesEnum(_messages.Enum):
        """Required. Immutable. Type of resource.

    Values:
      TYPE_UNSPECIFIED: Type not specified.
      STORAGE_BUCKET: Cloud Storage bucket.
      BIGQUERY_DATASET: BigQuery dataset.
    """
        TYPE_UNSPECIFIED = 0
        STORAGE_BUCKET = 1
        BIGQUERY_DATASET = 2
    name = _messages.StringField(1)
    readAccessMode = _messages.EnumField('ReadAccessModeValueValuesEnum', 2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)