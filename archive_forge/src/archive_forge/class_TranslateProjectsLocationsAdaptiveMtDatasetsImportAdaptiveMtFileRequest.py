from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsAdaptiveMtDatasetsImportAdaptiveMtFileRequest(_messages.Message):
    """A
  TranslateProjectsLocationsAdaptiveMtDatasetsImportAdaptiveMtFileRequest
  object.

  Fields:
    importAdaptiveMtFileRequest: A ImportAdaptiveMtFileRequest resource to be
      passed as the request body.
    parent: Required. The resource name of the file, in form of
      `projects/{project-number-or-
      id}/locations/{location_id}/adaptiveMtDatasets/{dataset}`
  """
    importAdaptiveMtFileRequest = _messages.MessageField('ImportAdaptiveMtFileRequest', 1)
    parent = _messages.StringField(2, required=True)