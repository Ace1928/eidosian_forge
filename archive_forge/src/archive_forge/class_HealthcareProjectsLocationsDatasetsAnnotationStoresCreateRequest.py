from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsAnnotationStoresCreateRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsAnnotationStoresCreateRequest
  object.

  Fields:
    annotationStore: A AnnotationStore resource to be passed as the request
      body.
    annotationStoreId: Required. The ID of the Annotation store that is being
      created. The string must match the following regex:
      `[\\p{L}\\p{N}_\\-\\.]{1,256}`.
    parent: Required. The name of the dataset this Annotation store belongs
      to.
  """
    annotationStore = _messages.MessageField('AnnotationStore', 1)
    annotationStoreId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)