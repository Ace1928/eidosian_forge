from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def UndeployIndexBeta(self, index_endpoint_ref, args):
    """Undeploy an index to an index endpoint."""
    undeploy_index_req = self.messages.GoogleCloudAiplatformV1beta1UndeployIndexRequest(deployedIndexId=args.deployed_index_id)
    request = self.messages.AiplatformProjectsLocationsIndexEndpointsUndeployIndexRequest(indexEndpoint=index_endpoint_ref.RelativeName(), googleCloudAiplatformV1beta1UndeployIndexRequest=undeploy_index_req)
    return self._service.UndeployIndex(request)