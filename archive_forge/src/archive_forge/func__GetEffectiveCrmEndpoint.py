from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from six.moves.urllib import parse
def _GetEffectiveCrmEndpoint(location):
    """Returns regional Tag Bindings Endpoint based on the regional location."""
    endpoint = apis.GetEffectiveApiEndpoint(CRM_API_NAME, CRM_API_VERSION)
    return _DeriveCrmRegionalEndpoint(endpoint, location)