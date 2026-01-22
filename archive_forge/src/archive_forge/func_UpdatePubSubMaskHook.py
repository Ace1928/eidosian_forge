from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_encoding
def UpdatePubSubMaskHook(unused_ref, args, req):
    """Constructs updateMask for patch requests of PubSub targets.

  Args:
    unused_ref: A resource ref to the parsed Job resource
    args: The parsed args namespace from CLI
    req: Created Patch request for the API call.

  Returns:
    Modified request for the API call.
  """
    pubsub_fields = {'--message-body': 'pubsubTarget.data', '--message-body-from-file': 'pubsubTarget.data', '--topic': 'pubsubTarget.topicName', '--clear-attributes': 'pubsubTarget.attributes', '--remove-attributes': 'pubsubTarget.attributes', '--update-attributes': 'pubsubTarget.attributes'}
    req.updateMask = _GenerateUpdateMask(args, pubsub_fields)
    return req