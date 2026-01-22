from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as compute_util
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.compute.sole_tenancy.node_groups import util
from six.moves import map
def UpdateShareSetting(self, node_group_ref, share_setting):
    """Sets the share setting on a node group."""
    share_setting_ref = util.BuildShareSettings(self.messages, share_setting)
    set_request = self.messages.NodeGroup(shareSettings=share_setting_ref)
    request = self.messages.ComputeNodeGroupsPatchRequest(nodeGroupResource=set_request, nodeGroup=node_group_ref.Name(), project=node_group_ref.project, zone=node_group_ref.zone)
    return self._service.Patch(request)