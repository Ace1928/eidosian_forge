from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
def ResolveExistingReservationArgs(args, resources):
    """Method to translate existing-reservations args into URLs."""
    resolver = compute_flags.ResourceResolver.FromMap('reservation', {compute_scope.ScopeEnum.ZONE: 'compute.reservations'})
    existing_reservations = getattr(args, 'existing_reservation', None)
    if existing_reservations is None:
        return []
    reservation_urls = []
    for reservation in existing_reservations:
        reservation_ref = resolver.ResolveResources([reservation['name']], compute_scope.ScopeEnum.ZONE, reservation['zone'], resources)[0]
        reservation_urls.append(reservation_ref.SelfLink())
    return reservation_urls