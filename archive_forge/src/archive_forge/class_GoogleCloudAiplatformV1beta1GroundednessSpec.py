from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1GroundednessSpec(_messages.Message):
    """Spec for groundedness metric.

  Fields:
    version: Optional. Which version to use for evaluation.
  """
    version = _messages.IntegerField(1, variant=_messages.Variant.INT32)