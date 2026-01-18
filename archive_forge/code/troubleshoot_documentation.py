from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.os_config import troubleshooter
Resolves the arguments into an instance.

    Args:
      holder: the api holder
      compute_client: the compute client
      args: The command line arguments.

    Returns:
      An instance reference to a VM.
    