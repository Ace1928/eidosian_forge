from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DicomFilterConfig(_messages.Message):
    """Specifies the filter configuration for DICOM resources.

  Fields:
    resourcePathsGcsUri: The Google Cloud Storage location of the filter
      configuration file. The `gcs_uri` must be in the format
      "gs://bucket/path/to/object" The filter configuration file must contain
      a list resource paths separated by new line characters (\\n or \\r\\n).
      Each resource path must be in the format
      "/studies/{studyUID}[/series/{seriesUID}[/instances/{instanceUID}]]" The
      Cloud Healthcare API service account must have the
      `roles/storage.objectViewer` Cloud IAM role for this Cloud Storage
      location.
  """
    resourcePathsGcsUri = _messages.StringField(1)