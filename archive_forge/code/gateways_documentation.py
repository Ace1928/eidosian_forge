from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.api_gateway import base
from googlecloudsdk.command_lib.api_gateway import common_flags
Creates a new gateway object.

    Args:
      gateway_ref: Resource, a resource reference for the gateway
      api_config: Resource, a resource reference for the gateway
      display_name: Optional display name
      labels: Optional cloud labels

    Returns:
      Long running operation.
    