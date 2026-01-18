from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import heapq
import sys
from containerregistry.client.v2_2 import docker_image
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import exceptions
This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Raises:
      ArgumentError: If the user provided the flag --show-occurrences-from but
        --show-occurrences=False.
      InvalidImageNameError: If the user specified an invalid image name.
    Returns:
      Some value that we want to have printed later.
    