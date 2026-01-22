from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _Await(result_service, operation_ref, message):
    client = GetAdminClient()
    poller = waiter.CloudOperationPoller(result_service, client.operations)
    return waiter.WaitFor(poller, operation_ref, message)