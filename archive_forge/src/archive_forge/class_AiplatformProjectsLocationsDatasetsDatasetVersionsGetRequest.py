from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDatasetsDatasetVersionsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsDatasetsDatasetVersionsGetRequest object.

  Fields:
    name: Required. The resource name of the Dataset version to delete.
      Format: `projects/{project}/locations/{location}/datasets/{dataset}/data
      setVersions/{dataset_version}`
    readMask: Mask specifying which fields to read.
  """
    name = _messages.StringField(1, required=True)
    readMask = _messages.StringField(2)