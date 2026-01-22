from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddOutputPathBigQueryArgs(parser):
    """Add BigQuery destination args to argument list."""
    bigquery_group = parser.add_group(mutex=False, required=False, help='The BigQuery destination for exporting assets.')
    resource = yaml_data.ResourceYAMLData.FromPath('bq.table')
    table_dic = resource.GetData()
    attributes = table_dic['attributes']
    for attr in attributes:
        if attr['attribute_name'] == 'dataset':
            attr['attribute_name'] = 'bigquery-dataset'
    arg_specs = [resource_args.GetResourcePresentationSpec(verb='export to', name='bigquery-table', required=True, prefixes=False, positional=False, resource_data=table_dic)]
    concept_parsers.ConceptParser(arg_specs).AddToParser(bigquery_group)
    base.Argument('--output-bigquery-force', action='store_true', dest='force_', default=False, required=False, help='If the destination table already exists and this flag is specified, the table will be overwritten by the contents of assets snapshot. If the flag is not specified and the destination table already exists, the export call returns an error.').AddToParser(bigquery_group)
    base.Argument('--per-asset-type', action='store_true', dest='per_type_', default=False, required=False, help='If the flag is specified, the snapshot results will be written to one or more tables, each of which contains results of one asset type.').AddToParser(bigquery_group)
    base.ChoiceArgument('--partition-key', required=False, choices=['read-time', 'request-time'], help_str='If specified. the snapshot results will be written to partitioned table(s) with two additional timestamp columns, readTime and requestTime, one of which will be the partition key.').AddToParser(bigquery_group)