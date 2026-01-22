from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstanceConfigsSsdCachesGetRequest(_messages.Message):
    """A SpannerProjectsInstanceConfigsSsdCachesGetRequest object.

  Fields:
    name: Required. The name of the requested SSD cache. Values are of the
      form `projects//instanceConfigs//ssdCaches/`.
  """
    name = _messages.StringField(1, required=True)