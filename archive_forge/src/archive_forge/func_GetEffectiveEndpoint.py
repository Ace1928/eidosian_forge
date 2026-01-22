from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves.urllib import parse
def GetEffectiveEndpoint(version, region, is_prediction=False):
    """Returns regional AI Platform endpoint, or raise an error if the region not set."""
    endpoint = apis.GetEffectiveApiEndpoint(constants.AI_PLATFORM_API_NAME, constants.AI_PLATFORM_API_VERSION[version])
    return DeriveAiplatformRegionalEndpoint(endpoint, region, is_prediction=is_prediction)