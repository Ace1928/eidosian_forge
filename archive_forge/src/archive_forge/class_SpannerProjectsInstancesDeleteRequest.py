from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDeleteRequest(_messages.Message):
    """A SpannerProjectsInstancesDeleteRequest object.

  Fields:
    name: Required. The name of the instance to be deleted. Values are of the
      form `projects//instances/`
  """
    name = _messages.StringField(1, required=True)