from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AwaitCluster(operation_ref, message):
    """Waits for cluster long running operation to complete."""
    client = GetAdminClient()
    return _Await(client.projects_instances_clusters, operation_ref, message)