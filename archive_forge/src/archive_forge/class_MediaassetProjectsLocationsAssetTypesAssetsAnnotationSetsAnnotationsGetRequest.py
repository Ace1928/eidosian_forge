from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsGetRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsGe
  tRequest object.

  Fields:
    name: Required. The name of the Annotation to retrieve. Format: `projects/
      {project}/locations/{location}/assetTypes/{asset_type}/assets/{asset}/an
      notationSets/{annotation_set}/annotations/{annotation}`
  """
    name = _messages.StringField(1, required=True)