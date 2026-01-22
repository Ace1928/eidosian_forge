from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import re
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves.urllib import parse
def GetEffectiveAssuredWorkloadsEndpoint(release_track, region):
    """Returns regional Assured Workloads endpoint, or global if region not set."""
    endpoint = apis.GetEffectiveApiEndpoint(util.API_NAME, util.GetApiVersion(release_track))
    if region:
        return DeriveAssuredWorkloadsRegionalEndpoint(endpoint, region)
    return endpoint