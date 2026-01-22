from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesGetRequest(_messages.Message):
    """A SpannerProjectsInstancesGetRequest object.

  Fields:
    fieldMask: If field_mask is present, specifies the subset of Instance
      fields that should be returned. If absent, all Instance fields are
      returned.
    name: Required. The name of the requested instance. Values are of the form
      `projects//instances/`.
  """
    fieldMask = _messages.StringField(1)
    name = _messages.StringField(2, required=True)