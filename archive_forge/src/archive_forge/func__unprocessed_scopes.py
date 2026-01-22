from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instances import exceptions
from googlecloudsdk.command_lib.compute.instances import flags
def _unprocessed_scopes(self, args, instance_ref, client):
    """Return scopes to set for the instance."""
    if args.no_scopes:
        return []
    if args.scopes is not None:
        return args.scopes
    return self._original_scopes(instance_ref, client)