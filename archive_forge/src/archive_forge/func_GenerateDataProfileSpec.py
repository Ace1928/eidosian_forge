from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.iam import iam_util
def GenerateDataProfileSpec(args):
    """Generate DataProfileSpec From Arguments."""
    module = dataplex_api.GetMessageModule()
    if args.IsSpecified('data_profile_spec_file'):
        dataprofilespec = dataplex_api.ReadObject(args.data_profile_spec_file)
        if dataprofilespec is not None:
            dataprofilespec = messages_util.DictToMessageWithErrorCheck(dataplex_api.SnakeToCamelDict(dataprofilespec), module.GoogleCloudDataplexV1DataProfileSpec)
    else:
        exclude_fields, include_fields, sampling_percent, row_filter = [None] * 4
        if hasattr(args, 'exclude_field_names') and args.IsSpecified('exclude_field_names'):
            exclude_fields = module.GoogleCloudDataplexV1DataProfileSpecSelectedFields(fieldNames=list((val.strip() for val in args.exclude_field_names.split(','))))
        if hasattr(args, 'include_field_names') and args.IsSpecified('include_field_names'):
            include_fields = module.GoogleCloudDataplexV1DataProfileSpecSelectedFields(fieldNames=list((val.strip() for val in args.include_field_names.split(','))))
        if hasattr(args, 'sampling_percent') and args.IsSpecified('sampling_percent'):
            sampling_percent = float(args.sampling_percent)
        if hasattr(args, 'row_filter') and args.IsSpecified('row_filter'):
            row_filter = args.row_filter
        dataprofilespec = module.GoogleCloudDataplexV1DataProfileSpec(excludeFields=exclude_fields, includeFields=include_fields, samplingPercent=sampling_percent, rowFilter=row_filter)
        if hasattr(args, 'export_results_table') and args.IsSpecified('export_results_table'):
            dataprofilespec.postScanActions = module.GoogleCloudDataplexV1DataProfileSpecPostScanActions(bigqueryExport=module.GoogleCloudDataplexV1DataProfileSpecPostScanActionsBigQueryExport(resultsTable=args.export_results_table))
    return dataprofilespec