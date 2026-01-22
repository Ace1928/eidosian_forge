from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnnotationChange(_messages.Message):
    """Specifies the annotation changes that should trigger notifications.

  Fields:
    annotationSetId: Required. Notifies for changes to any annotation nested
      under the given annotationSet of the parent AssetType.
  """
    annotationSetId = _messages.StringField(1)