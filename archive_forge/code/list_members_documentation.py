from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.services import common_flags
from googlecloudsdk.core import properties
Run command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Resource name and its parent name.
    