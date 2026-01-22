from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1IntegratedGradientsAttribution(_messages.Message):
    """Attributes credit by computing the Aumann-Shapley value taking advantage
  of the model's fully differentiable structure. Refer to this paper for more
  details: https://arxiv.org/abs/1703.01365

  Fields:
    blurBaselineConfig: Config for IG with blur baseline. When enabled, a
      linear path from the maximally blurred image to the input image is
      created. Using a blurred baseline instead of zero (black image) is
      motivated by the BlurIG approach explained here:
      https://arxiv.org/abs/2004.03383
    numIntegralSteps: Number of steps for approximating the path integral. A
      good value to start is 50 and gradually increase until the sum to diff
      property is met within the desired error range.
    smoothGradConfig: Config for SmoothGrad approximation of gradients. When
      enabled, the gradients are approximated by averaging the gradients from
      noisy samples in the vicinity of the inputs. Adding noise can help
      improve the computed gradients, see here for why:
      https://arxiv.org/pdf/1706.03825.pdf
  """
    blurBaselineConfig = _messages.MessageField('GoogleCloudMlV1BlurBaselineConfig', 1)
    numIntegralSteps = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    smoothGradConfig = _messages.MessageField('GoogleCloudMlV1SmoothGradConfig', 3)