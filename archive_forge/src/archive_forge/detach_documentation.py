from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import command
from googlecloudsdk.command_lib.container.fleet.policycontroller import flags
Sets the membership spec to DETACHED.

    Args:
      spec: The spec to be detached.

    Returns:
      Updated spec, based on the message api version.
    