from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def FilterTPUVMNodes(response, args):
    """Removes Cloud TPU V1 API nodes from the 'list' output.

  Used with 'compute tpus tpu-vm list'.

  Args:
    response: response to ListNodes.
    args: the arguments for the list command.

  Returns:
    A response with only TPU VM (non-V1 API) nodes.
  """
    del args
    return list(six.moves.filter(IsTPUVMNode, response))