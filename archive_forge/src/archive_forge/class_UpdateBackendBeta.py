from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_services_utils
from googlecloudsdk.command_lib.compute.backend_services import flags
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class UpdateBackendBeta(UpdateBackend):
    """Update an existing backend in a backend service.

  *{command}* updates a backend that is part of a backend
  service. This is useful for changing the way a backend
  behaves. Example changes that can be made include changing the
  load balancing policy and draining a backend by setting
  its capacity scaler to zero.

  Backends are instance groups or network endpoint groups. One
  of the `--network-endpoint-group` or `--instance-group` flags is required to
  identify the backend that you are modifying. You cannot change
  the instance group or network endpoint group associated with a backend, but
  you can remove a backend and add a new one with `backend-services
  remove-backend` and `backend-services add-backend`.

  The `gcloud compute backend-services edit` command can also
  update a backend if the use of a text editor is desired.

  For more information about the available settings, see
  https://cloud.google.com/load-balancing/docs/backend-service.
  """
    support_preference = True

    @classmethod
    def Args(cls, parser):
        flags.GLOBAL_REGIONAL_BACKEND_SERVICE_ARG.AddArgument(parser)
        flags.AddInstanceGroupAndNetworkEndpointGroupArgs(parser, 'update in')
        backend_flags.AddDescription(parser)
        backend_flags.AddBalancingMode(parser)
        backend_flags.AddCapacityLimits(parser)
        backend_flags.AddCapacityScalar(parser)
        backend_flags.AddFailover(parser, default=None)
        backend_flags.AddPreference(parser)

    def _ValidateArgs(self, args):
        """Overrides."""
        if not any([args.description is not None, args.balancing_mode, args.max_utilization is not None, args.max_rate is not None, args.max_rate_per_instance is not None, args.max_rate_per_endpoint is not None, args.max_connections is not None, args.max_connections_per_instance is not None, args.max_connections_per_endpoint is not None, args.capacity_scaler is not None, args.failover is not None, args.preference is not None]):
            raise exceptions.UpdatePropertyError('At least one property must be modified.')