from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V1beta1RemoveVisibilityLabelsResponse(_messages.Message):
    """Response message for the `RemoveVisibilityLabels` method. This response
  message is assigned to the `response` field of the returned Operation when
  that operation is done.

  Fields:
    labels: The updated set of visibility labels for this consumer on this
      service.
  """
    labels = _messages.StringField(1, repeated=True)