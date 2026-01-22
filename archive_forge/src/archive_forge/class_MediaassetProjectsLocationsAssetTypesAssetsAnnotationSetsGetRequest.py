from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsGetRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsGetRequest
  object.

  Fields:
    name: Required. The name of the AnnotationSet to retrieve. Format: `projec
      ts/{project}/locations/{location}/assetTypes/{asset_type}/assets/{asset}
      /annotationSets/{annotation_set}`
  """
    name = _messages.StringField(1, required=True)