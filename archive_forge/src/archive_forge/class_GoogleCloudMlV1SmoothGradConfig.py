from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1SmoothGradConfig(_messages.Message):
    """Config for SmoothGrad approximation of gradients. When enabled, the
  gradients are approximated by averaging the gradients from noisy samples in
  the vicinity of the inputs. Adding noise can help improve the computed
  gradients. See here for why https://arxiv.org/pdf/1706.03825.pdf

  Fields:
    featureNoiseSigma: Alternatively, set this to use different noise_sigma
      per feature. One entry per feature. No noise is added to features that
      are not set.
    noiseSigma: If set, this std. deviation will be used to apply noise to all
      features.
    noisySampleCount: The number of gradient samples to use for approximation.
      The higher this number, the more accurate the gradient is, but the
      runtime complexity of IG increases by this factor as well.
  """
    featureNoiseSigma = _messages.MessageField('GoogleCloudMlV1FeatureNoiseSigma', 1)
    noiseSigma = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    noisySampleCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)