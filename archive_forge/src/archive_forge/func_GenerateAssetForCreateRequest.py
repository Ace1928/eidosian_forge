from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.command_lib.iam import iam_util
def GenerateAssetForCreateRequest(args):
    """Create Asset for Message Create Requests."""
    module = dataplex_api.GetMessageModule()
    resource_spec_field = module.GoogleCloudDataplexV1AssetResourceSpec
    resource_spec = module.GoogleCloudDataplexV1AssetResourceSpec(name=args.resource_name, type=resource_spec_field.TypeValueValuesEnum(args.resource_type))
    if args.IsSpecified('resource_read_access_mode'):
        resource_spec.readAccessMode = resource_spec_field.ReadAccessModeValueValuesEnum(args.resource_read_access_mode)
    request = module.GoogleCloudDataplexV1Asset(description=args.description, displayName=args.display_name, labels=dataplex_api.CreateLabels(module.GoogleCloudDataplexV1Asset, args), resourceSpec=resource_spec)
    discovery = GenerateDiscoverySpec(args)
    if discovery != module.GoogleCloudDataplexV1AssetDiscoverySpec():
        setattr(request, 'discoverySpec', discovery)
    return request