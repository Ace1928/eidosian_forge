from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.api_lib.container import util as container_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as composer_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.
    