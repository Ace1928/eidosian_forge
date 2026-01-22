from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1SampledShapleyAttribution(_messages.Message):
    """An attribution method that approximates Shapley values for features that
  contribute to the label being predicted. A sampling strategy is used to
  approximate the value rather than considering all subsets of features.

  Fields:
    numPaths: The number of feature permutations to consider when
      approximating the Shapley values.
  """
    numPaths = _messages.IntegerField(1, variant=_messages.Variant.INT32)