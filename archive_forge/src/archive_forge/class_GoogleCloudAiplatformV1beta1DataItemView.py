from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1DataItemView(_messages.Message):
    """A container for a single DataItem and Annotations on it.

  Fields:
    annotations: The Annotations on the DataItem. If too many Annotations
      should be returned for the DataItem, this field will be truncated per
      annotations_limit in request. If it was, then the
      has_truncated_annotations will be set to true.
    dataItem: The DataItem.
    hasTruncatedAnnotations: True if and only if the Annotations field has
      been truncated. It happens if more Annotations for this DataItem met the
      request's annotation_filter than are allowed to be returned by
      annotations_limit. Note that if Annotations field is not being returned
      due to field mask, then this field will not be set to true no matter how
      many Annotations are there.
  """
    annotations = _messages.MessageField('GoogleCloudAiplatformV1beta1Annotation', 1, repeated=True)
    dataItem = _messages.MessageField('GoogleCloudAiplatformV1beta1DataItem', 2)
    hasTruncatedAnnotations = _messages.BooleanField(3)