from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.command_lib.iam import iam_util
def GenerateZoneForUpdateRequest(args):
    """Create Zone for Message Update Requests."""
    module = dataplex_api.GetMessageModule()
    return module.GoogleCloudDataplexV1Zone(description=args.description, displayName=args.display_name, labels=dataplex_api.CreateLabels(module.GoogleCloudDataplexV1Zone, args), discoverySpec=GenerateDiscoverySpec(args))