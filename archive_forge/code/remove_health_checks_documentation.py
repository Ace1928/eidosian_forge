from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.http_health_checks import (
from googlecloudsdk.command_lib.compute.target_pools import flags
Remove an HTTP health check from a target pool.

  *{command}* is used to remove an HTTP health check
  from a target pool. Health checks are used to determine
  the health status of instances in the target pool. For more
  information on health checks and load balancing, see
  [](https://cloud.google.com/compute/docs/load-balancing-and-autoscaling/)
  