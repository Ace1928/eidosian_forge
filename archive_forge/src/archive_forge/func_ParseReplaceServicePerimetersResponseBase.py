from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.accesscontextmanager import acm_printer
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import levels
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def ParseReplaceServicePerimetersResponseBase(lro, version):
    """Parse the Long Running Operation response of the ReplaceServicePerimeters call.

  Args:
    lro: Long Running Operation response of ReplaceServicePerimeters.
    version: version of the API. e.g. 'v1beta', 'v1'.

  Returns:
    The replacement Service Perimeters created by the ReplaceServicePerimeters
    call.

  Raises:
    ParseResponseError: if the response could not be parsed into the proper
    object.
  """
    client = util.GetClient(version=version)
    operation_ref = resources.REGISTRY.Parse(lro.name, collection='accesscontextmanager.operations')
    poller = common.BulkAPIOperationPoller(client.accessPolicies_servicePerimeters, client.operations, operation_ref)
    return waiter.WaitFor(poller, operation_ref, 'Waiting for Replace Service Perimeters operation [{}]'.format(operation_ref.Name()))