from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_session
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
import six
This is what is called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Raises:
      util.InvalidImageNameError: If the user specified an invalid
      (or non-existent) image name.
    Returns:
      A list of the deleted docker_name.Tag objects
    