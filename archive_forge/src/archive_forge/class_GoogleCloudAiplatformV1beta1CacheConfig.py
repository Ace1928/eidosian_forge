from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1CacheConfig(_messages.Message):
    """Config of GenAI caching features. This is a singleton resource.

  Fields:
    disableCache: If set to true, disables GenAI caching. Otherwise caching is
      enabled.
    name: Identifier. Name of the cache config. Format: -
      `projects/{project}/cacheConfig`.
  """
    disableCache = _messages.BooleanField(1)
    name = _messages.StringField(2)