from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1IdMatcher(_messages.Message):
    """Matcher for Features of an EntityType by Feature ID.

  Fields:
    ids: Required. The following are accepted as `ids`: * A single-element
      list containing only `*`, which selects all Features in the target
      EntityType, or * A list containing only Feature IDs, which selects only
      Features with those IDs in the target EntityType.
  """
    ids = _messages.StringField(1, repeated=True)