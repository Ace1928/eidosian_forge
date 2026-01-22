from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsAdaptiveMtDatasetsAdaptiveMtFilesGetRequest(_messages.Message):
    """A TranslateProjectsLocationsAdaptiveMtDatasetsAdaptiveMtFilesGetRequest
  object.

  Fields:
    name: Required. The resource name of the file, in form of
      `projects/{project-number-or-id}/locations/{location_id}/adaptiveMtDatas
      ets/{dataset}/adaptiveMtFiles/{file}`
  """
    name = _messages.StringField(1, required=True)