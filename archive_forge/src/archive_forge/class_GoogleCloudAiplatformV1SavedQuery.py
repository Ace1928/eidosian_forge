from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SavedQuery(_messages.Message):
    """A SavedQuery is a view of the dataset. It references a subset of
  annotations by problem type and filters.

  Fields:
    annotationFilter: Output only. Filters on the Annotations in the dataset.
    annotationSpecCount: Output only. Number of AnnotationSpecs in the context
      of the SavedQuery.
    createTime: Output only. Timestamp when this SavedQuery was created.
    displayName: Required. The user-defined name of the SavedQuery. The name
      can be up to 128 characters long and can consist of any UTF-8
      characters.
    etag: Used to perform a consistent read-modify-write update. If not set, a
      blind "overwrite" update happens.
    metadata: Some additional information about the SavedQuery.
    name: Output only. Resource name of the SavedQuery.
    problemType: Required. Problem type of the SavedQuery. Allowed values: *
      IMAGE_CLASSIFICATION_SINGLE_LABEL * IMAGE_CLASSIFICATION_MULTI_LABEL *
      IMAGE_BOUNDING_POLY * IMAGE_BOUNDING_BOX *
      TEXT_CLASSIFICATION_SINGLE_LABEL * TEXT_CLASSIFICATION_MULTI_LABEL *
      TEXT_EXTRACTION * TEXT_SENTIMENT * VIDEO_CLASSIFICATION *
      VIDEO_OBJECT_TRACKING
    supportAutomlTraining: Output only. If the Annotations belonging to the
      SavedQuery can be used for AutoML training.
    updateTime: Output only. Timestamp when SavedQuery was last updated.
  """
    annotationFilter = _messages.StringField(1)
    annotationSpecCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    createTime = _messages.StringField(3)
    displayName = _messages.StringField(4)
    etag = _messages.StringField(5)
    metadata = _messages.MessageField('extra_types.JsonValue', 6)
    name = _messages.StringField(7)
    problemType = _messages.StringField(8)
    supportAutomlTraining = _messages.BooleanField(9)
    updateTime = _messages.StringField(10)