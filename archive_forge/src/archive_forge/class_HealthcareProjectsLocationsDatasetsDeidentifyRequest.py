from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDeidentifyRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsDeidentifyRequest object.

  Fields:
    deidentifyDatasetRequest: A DeidentifyDatasetRequest resource to be passed
      as the request body.
    sourceDataset: Required. Source dataset resource name. For example,
      `projects/{project_id}/locations/{location_id}/datasets/{dataset_id}`.
  """
    deidentifyDatasetRequest = _messages.MessageField('DeidentifyDatasetRequest', 1)
    sourceDataset = _messages.StringField(2, required=True)