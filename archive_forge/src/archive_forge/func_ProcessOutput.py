from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.media.asset import utils
from googlecloudsdk.core import resources
def ProcessOutput(response, args):
    """Wait for operations to finish and return the resource."""
    api_version = utils.GetApiVersionFromArgs(args)
    utils.WaitForOperation(response, api_version)
    project = utils.GetProject()
    location = utils.GetLocation(args)
    resource_ref = resources.REGISTRY.Create('mediaasset.projects.locations.assetTypes.assets', projectsId=project, locationsId=location, assetTypesId=args.asset_type, assetsId=args.asset)
    if 'delete' in args.command_path:
        return response
    request_message = utils.GetApiMessage(api_version).MediaassetProjectsLocationsAssetTypesAssetsGetRequest(name=resource_ref.RelativeName())
    return GetExistingResource(api_version, request_message)