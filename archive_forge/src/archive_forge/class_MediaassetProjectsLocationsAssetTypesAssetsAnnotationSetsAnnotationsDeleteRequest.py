from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsDeleteRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsDe
  leteRequest object.

  Fields:
    etag: The current etag of the annotation. If an etag is provided and does
      not match the current etag of the annotation, deletion will be blocked
      and a FAILED_PRECONDITION error will be returned.
    name: Required. The name of the annotation to delete. Format: `projects/{p
      roject}/locations/{location}/assetTypes/{asset_type}/assets/{asset}/anno
      tationSets/{annotation_set}/annotations/{annotation}`
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)