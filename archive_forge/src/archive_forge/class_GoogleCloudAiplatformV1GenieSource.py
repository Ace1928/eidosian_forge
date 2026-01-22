from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1GenieSource(_messages.Message):
    """Contains information about the source of the models generated from
  Generative AI Studio.

  Fields:
    baseModelUri: Required. The public base model URI.
  """
    baseModelUri = _messages.StringField(1)