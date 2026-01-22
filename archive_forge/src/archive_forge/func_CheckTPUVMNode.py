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
def CheckTPUVMNode(response, args):
    """Verifies that the node is a TPU VM node.

  If it is not a TPU VM node, exit with an error instead.

  Args:
    response: response to GetNode.
    args: the arguments for the list command.

  Returns:
    The response to GetNode if the node is TPU VM.
  """
    del args
    if IsTPUVMNode(response):
        return response
    log.err.Print('ERROR: Please use "gcloud compute tpus describe" for Cloud TPU nodes that are not TPU VM.')
    sys.exit(1)