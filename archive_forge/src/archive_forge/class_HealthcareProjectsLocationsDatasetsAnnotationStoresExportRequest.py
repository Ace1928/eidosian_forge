from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsAnnotationStoresExportRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsAnnotationStoresExportRequest
  object.

  Fields:
    exportAnnotationsRequest: A ExportAnnotationsRequest resource to be passed
      as the request body.
    name: Required. The name of the Annotation store to export annotations to,
      in the format of `projects/{project_id}/locations/{location_id}/datasets
      /{dataset_id}/annotationStores/{annotation_store_id}`.
  """
    exportAnnotationsRequest = _messages.MessageField('ExportAnnotationsRequest', 1)
    name = _messages.StringField(2, required=True)