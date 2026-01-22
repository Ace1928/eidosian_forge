from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.core.util import times
import six
def MakeReservationMessageFromArgs(messages, args, reservation_ref, resources):
    """Construct reservation message from args passed in."""
    accelerators = MakeGuestAccelerators(messages, getattr(args, 'accelerator', None))
    local_ssds = MakeLocalSsds(messages, getattr(args, 'local_ssd', None))
    share_settings = MakeShareSettingsWithArgs(messages, args, getattr(args, 'share_setting', None))
    source_instance_template_ref = ResolveSourceInstanceTemplate(args, resources) if args.IsKnownAndSpecified('source_instance_template') else None
    specific_reservation = MakeSpecificSKUReservationMessage(messages, args.vm_count, accelerators, local_ssds, args.machine_type, args.min_cpu_platform, getattr(args, 'location_hint', None), getattr(args, 'maintenance_freeze_duration', None), getattr(args, 'maintenance_interval', None), source_instance_template_ref)
    resource_policies = MakeResourcePolicies(messages, reservation_ref, getattr(args, 'resource_policies', None), resources)
    return MakeReservationMessage(messages, reservation_ref.Name(), share_settings, specific_reservation, resource_policies, args.require_specific_reservation, reservation_ref.zone, getattr(args, 'delete_at_time', None), getattr(args, 'delete_after_duration', None))