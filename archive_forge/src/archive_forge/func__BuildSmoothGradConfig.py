from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.ai import operations
from googlecloudsdk.api_lib.ai.models import client
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import models_util
from googlecloudsdk.command_lib.ai import operations_util
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.core import yaml
def _BuildSmoothGradConfig(self, args):
    if args.smooth_grad_noise_sigma is None and args.smooth_grad_noisy_sample_count is None and (args.smooth_grad_noise_sigma_by_feature is None):
        return None
    if args.smooth_grad_noise_sigma is not None and args.smooth_grad_noise_sigma_by_feature is not None:
        raise gcloud_exceptions.BadArgumentException('--smooth-grad-noise-sigma', 'Only one of smooth-grad-noise-sigma and smooth-grad-noise-sigma-by-feature can be set.')
    smooth_grad_config = self.messages.GoogleCloudAiplatformV1beta1SmoothGradConfig(noiseSigma=args.smooth_grad_noise_sigma, noisySampleCount=args.smooth_grad_noisy_sample_count)
    sigmas = args.smooth_grad_noise_sigma_by_feature
    if sigmas:
        smooth_grad_config.featureNoiseSigma = self.messages.GoogleCloudAiplatformV1beta1FeatureNoiseSigma(noiseSigma=[self.messages.GoogleCloudAiplatformV1beta1FeatureNoiseSigmaNoiseSigmaForFeature(name=k, sigma=float(sigmas[k])) for k in sigmas])
    return smooth_grad_config