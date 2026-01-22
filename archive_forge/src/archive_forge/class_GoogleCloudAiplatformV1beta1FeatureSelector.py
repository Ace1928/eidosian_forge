from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeatureSelector(_messages.Message):
    """Selector for Features of an EntityType.

  Fields:
    idMatcher: Required. Matches Features based on ID.
  """
    idMatcher = _messages.MessageField('GoogleCloudAiplatformV1beta1IdMatcher', 1)