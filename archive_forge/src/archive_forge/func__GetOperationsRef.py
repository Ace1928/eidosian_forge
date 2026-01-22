from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as compute_util
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.compute.sole_tenancy.node_groups import util
from six.moves import map
def _GetOperationsRef(self, operation):
    return self.resources.Parse(operation.selfLink, collection='compute.zoneOperations')