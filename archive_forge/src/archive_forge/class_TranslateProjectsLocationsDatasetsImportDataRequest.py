from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsDatasetsImportDataRequest(_messages.Message):
    """A TranslateProjectsLocationsDatasetsImportDataRequest object.

  Fields:
    dataset: Required. Name of the dataset. In form of `projects/{project-
      number-or-id}/locations/{location-id}/datasets/{dataset-id}`
    importDataRequest: A ImportDataRequest resource to be passed as the
      request body.
  """
    dataset = _messages.StringField(1, required=True)
    importDataRequest = _messages.MessageField('ImportDataRequest', 2)