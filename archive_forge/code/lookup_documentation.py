from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastream import stream_objects
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.datastream import resource_args
from googlecloudsdk.command_lib.datastream.objects import flags as so_flags
from googlecloudsdk.core import properties
Lookup a Datastream stream object.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      A dict object representing the looked up stream object if the lookup was
      successful.
    