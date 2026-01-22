from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta3Dataset(_messages.Message):
    """A singleton resource under a Processor which configures a collection of
  documents.

  Enums:
    StateValueValuesEnum: Required. State of the dataset. Ignored when
      updating dataset.

  Fields:
    documentWarehouseConfig: Optional. Deprecated. Warehouse-based dataset
      configuration is not supported.
    gcsManagedConfig: Optional. User-managed Cloud Storage dataset
      configuration. Use this configuration if the dataset documents are
      stored under a user-managed Cloud Storage location.
    name: Dataset resource name. Format:
      `projects/{project}/locations/{location}/processors/{processor}/dataset`
    spannerIndexingConfig: Optional. A lightweight indexing source with low
      latency and high reliability, but lacking advanced features like CMEK
      and content-based search.
    state: Required. State of the dataset. Ignored when updating dataset.
    unmanagedDatasetConfig: Optional. Unmanaged dataset configuration. Use
      this configuration if the dataset documents are managed by the document
      service internally (not user-managed).
  """

    class StateValueValuesEnum(_messages.Enum):
        """Required. State of the dataset. Ignored when updating dataset.

    Values:
      STATE_UNSPECIFIED: Default unspecified enum, should not be used.
      UNINITIALIZED: Dataset has not been initialized.
      INITIALIZING: Dataset is being initialized.
      INITIALIZED: Dataset has been initialized.
    """
        STATE_UNSPECIFIED = 0
        UNINITIALIZED = 1
        INITIALIZING = 2
        INITIALIZED = 3
    documentWarehouseConfig = _messages.MessageField('GoogleCloudDocumentaiV1beta3DatasetDocumentWarehouseConfig', 1)
    gcsManagedConfig = _messages.MessageField('GoogleCloudDocumentaiV1beta3DatasetGCSManagedConfig', 2)
    name = _messages.StringField(3)
    spannerIndexingConfig = _messages.MessageField('GoogleCloudDocumentaiV1beta3DatasetSpannerIndexingConfig', 4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    unmanagedDatasetConfig = _messages.MessageField('GoogleCloudDocumentaiV1beta3DatasetUnmanagedDatasetConfig', 6)