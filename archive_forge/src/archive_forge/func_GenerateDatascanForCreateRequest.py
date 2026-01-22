from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.iam import iam_util
def GenerateDatascanForCreateRequest(args):
    """Create Datascan for Message Create Requests."""
    module = dataplex_api.GetMessageModule()
    request = module.GoogleCloudDataplexV1DataScan(description=args.description, displayName=args.display_name, labels=dataplex_api.CreateLabels(module.GoogleCloudDataplexV1DataScan, args), data=GenerateData(args), executionSpec=GenerateExecutionSpecForCreateRequest(args))
    if args.scan_type == 'PROFILE':
        if hasattr(args, 'data_quality_spec_file') and args.IsSpecified('data_quality_spec_file'):
            raise ValueError('Data quality spec file specified for data profile scan.')
        else:
            request.dataProfileSpec = GenerateDataProfileSpec(args)
    elif args.scan_type == 'QUALITY':
        if hasattr(args, 'data_profile_spec_file') and args.IsSpecified('data_profile_spec_file'):
            raise ValueError('Data profile spec file specified for data quality scan.')
        elif args.IsSpecified('data_quality_spec_file'):
            request.dataQualitySpec = GenerateDataQualitySpec(args)
        else:
            raise ValueError('If scan-type="QUALITY" , data-quality-spec-file is a required argument.')
    return request