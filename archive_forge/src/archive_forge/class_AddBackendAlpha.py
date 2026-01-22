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
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class AddBackendAlpha(AddBackend):
    """Add a backend to a backend service.

  *{command}* adds a backend to a Google Cloud load balancer or Traffic
  Director. Depending on the load balancing scheme of the backend service,
  backends can be instance groups (managed or unmanaged), zonal network endpoint
  groups (zonal NEGs), serverless NEGs, or an internet NEG. For more
  information, see the [backend services
  overview](https://cloud.google.com/load-balancing/docs/backend-service).

  For most load balancers, you can define how Google Cloud measures capacity by
  selecting a balancing mode. For more information, see [traffic
  distribution](https://cloud.google.com/load-balancing/docs/backend-service#traffic_distribution).

  To modify a backend, use the `gcloud compute backend-services update-backend`
  or `gcloud compute backend-services edit` command.
  """
    support_preference = True