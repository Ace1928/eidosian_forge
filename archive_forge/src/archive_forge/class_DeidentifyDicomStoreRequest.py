from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeidentifyDicomStoreRequest(_messages.Message):
    """Creates a new DICOM store with sensitive information de-identified.

  Fields:
    config: Deidentify configuration. Only one of `config` and
      `gcs_config_uri` can be specified.
    destinationStore: Required. The name of the DICOM store to create and
      write the redacted data to. For example, `projects/{project_id}/location
      s/{location_id}/datasets/{dataset_id}/dicomStores/{dicom_store_id}`. *
      The destination dataset must exist. * The source dataset and destination
      dataset must both reside in the same location. De-identifying data
      across multiple locations is not supported. * The destination DICOM
      store must not exist. * The caller must have the necessary permissions
      to create the destination DICOM store.
  """
    config = _messages.MessageField('DeidentifyConfig', 1)
    destinationStore = _messages.StringField(2)