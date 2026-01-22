from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppengineV1betaLocationMetadata(_messages.Message):
    """Metadata for the given google.cloud.location.Location.

  Fields:
    flexibleEnvironmentAvailable: App Engine flexible environment is available
      in the given location.@OutputOnly
    searchApiAvailable: Output only. Search API
      (https://cloud.google.com/appengine/docs/standard/python/search) is
      available in the given location.
    standardEnvironmentAvailable: App Engine standard environment is available
      in the given location.@OutputOnly
  """
    flexibleEnvironmentAvailable = _messages.BooleanField(1)
    searchApiAvailable = _messages.BooleanField(2)
    standardEnvironmentAvailable = _messages.BooleanField(3)