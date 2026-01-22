from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeidentifyDatasetRequest(_messages.Message):
    """Redacts identifying information from the specified dataset.

  Fields:
    config: Deidentify configuration. Only one of `config` and
      `gcs_config_uri` can be specified.
    destinationDataset: Required. The name of the dataset resource to create
      and write the redacted data to. For example,
      `projects/{project_id}/locations/{location_id}/datasets/{dataset_id}`. *
      The destination dataset must not exist. * The destination dataset must
      be in the same location as the source dataset. De-identifying data
      across multiple locations is not supported.
  """
    config = _messages.MessageField('DeidentifyConfig', 1)
    destinationDataset = _messages.StringField(2)