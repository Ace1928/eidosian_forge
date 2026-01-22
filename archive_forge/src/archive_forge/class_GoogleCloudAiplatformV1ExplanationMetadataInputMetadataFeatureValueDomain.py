from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExplanationMetadataInputMetadataFeatureValueDomain(_messages.Message):
    """Domain details of the input feature value. Provides numeric information
  about the feature, such as its range (min, max). If the feature has been
  pre-processed, for example with z-scoring, then it provides information
  about how to recover the original feature. For example, if the input feature
  is an image and it has been pre-processed to obtain 0-mean and stddev = 1
  values, then original_mean, and original_stddev refer to the mean and stddev
  of the original feature (e.g. image tensor) from which input feature (with
  mean = 0 and stddev = 1) was obtained.

  Fields:
    maxValue: The maximum permissible value for this feature.
    minValue: The minimum permissible value for this feature.
    originalMean: If this input feature has been normalized to a mean value of
      0, the original_mean specifies the mean value of the domain prior to
      normalization.
    originalStddev: If this input feature has been normalized to a standard
      deviation of 1.0, the original_stddev specifies the standard deviation
      of the domain prior to normalization.
  """
    maxValue = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    minValue = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    originalMean = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    originalStddev = _messages.FloatField(4, variant=_messages.Variant.FLOAT)