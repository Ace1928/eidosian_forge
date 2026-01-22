from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _ValidateGroupMatchesArgs(args):
    """Validate if the group arg is used with the correct group specific flags."""
    invalid_arg = None
    if args.instance_group:
        if args.max_rate_per_endpoint is not None:
            invalid_arg = '--max-rate-per-endpoint'
        elif args.max_connections_per_endpoint is not None:
            invalid_arg = '--max-connections-per-endpoint'
        if invalid_arg is not None:
            raise exceptions.InvalidArgumentException(invalid_arg, 'cannot be set with --instance-group')
    elif args.network_endpoint_group:
        if args.max_rate_per_instance is not None:
            invalid_arg = '--max-rate-per-instance'
        elif args.max_connections_per_instance is not None:
            invalid_arg = '--max-connections-per-instance'
        if invalid_arg is not None:
            raise exceptions.InvalidArgumentException(invalid_arg, 'cannot be set with --network-endpoint-group')