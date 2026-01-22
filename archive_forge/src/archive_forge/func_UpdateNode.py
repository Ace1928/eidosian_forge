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
def UpdateNode(self, name, zone, node, update_mask, poller_message):
    """Updates the TPU node in the given zone."""
    project = properties.VALUES.core.project.Get(required=True)
    fully_qualified_node_name_ref = resources.REGISTRY.Parse(name, params={'locationsId': zone, 'projectsId': project}, collection='tpu.projects.locations.nodes')
    request = self.messages.TpuProjectsLocationsNodesPatchRequest(name=fully_qualified_node_name_ref.RelativeName(), node=node, updateMask=update_mask)
    operation = self.client.projects_locations_nodes.Patch(request)
    operation_ref = resources.REGISTRY.ParseRelativeName(operation.name, collection='tpu.projects.locations.operations')
    return self.WaitForOperation(operation_ref, poller_message)