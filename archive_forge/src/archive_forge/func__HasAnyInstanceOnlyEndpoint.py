from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.util.apis import arg_utils
def _HasAnyInstanceOnlyEndpoint(self, endpoints):
    """Checks if endpoint list has an endpoint with instance only."""
    for arg_endpoint in endpoints:
        if 'instance' in arg_endpoint and len(arg_endpoint) == 1:
            return True
    return False