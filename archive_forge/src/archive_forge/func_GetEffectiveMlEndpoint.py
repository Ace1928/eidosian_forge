from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves.urllib import parse
def GetEffectiveMlEndpoint(region):
    """Returns regional ML Endpoint, or global if region not set."""
    endpoint = apis.GetEffectiveApiEndpoint(ML_API_NAME, ML_API_VERSION)
    if region and region != 'global':
        return DeriveMLRegionalEndpoint(endpoint, region)
    return endpoint