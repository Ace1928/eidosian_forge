from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.command_lib.dataflow import job_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def ExtractOutputs(args):
    """Parses outputs from args, returning a JSON string with the results."""
    outputs = []
    if args.bigquery_table:
        bq_project = None
        dataset = None
        table = None
        table_parts = args.bigquery_table.split('.')
        if len(table_parts) == 3:
            bq_project, dataset, table = table_parts
        elif len(table_parts) == 2:
            dataset, table = table_parts
        elif len(table_parts) == 1:
            table, = table_parts
        else:
            raise exceptions.InvalidArgumentException('--bigquery-table', 'Malformed table identifier. Use format "project.dataset.table".')
        if bq_project is None:
            bq_project = args.bigquery_project if args.bigquery_project else properties.VALUES.core.project.GetOrFail()
        elif args.bigquery_project and args.bigquery_project != bq_project:
            raise exceptions.InvalidArgumentException('--bigquery-project', '"{}" does not match project "{}" set in qualified `--bigquery-table`.'.format(args.bigquery_project, bq_project))
        if dataset is None:
            if not args.bigquery_dataset:
                raise exceptions.RequiredArgumentException('--bigquery-dataset', 'Must be specified when `--bigquery-table` is unqualified.')
            dataset = args.bigquery_dataset
        elif args.bigquery_dataset and args.bigquery_dataset != dataset:
            raise exceptions.InvalidArgumentException('--bigquery-dataset', '"{}" does not match dataset "{}" set in qualified `--bigquery-table`.'.format(args.bigquery_dataset, dataset))
        table_config = collections.OrderedDict([('projectId', bq_project), ('datasetId', dataset), ('tableId', table)])
        write_disposition = {'write-empty': 'WRITE_EMPTY', 'write-truncate': 'WRITE_TRUNCATE', 'write-append': 'WRITE_APPEND'}[args.bigquery_write_disposition]
        bq_config = collections.OrderedDict([('type', 'bigquery'), ('table', table_config), ('writeDisposition', write_disposition)])
        outputs.append(bq_config)
    if args.pubsub_topic:
        create_disposition = {'create-if-not-found': 'CREATE_IF_NOT_FOUND', 'fail-if-not-found': 'FAIL_IF_NOT_FOUND'}[args.pubsub_create_disposition]
        pubsub_config = collections.OrderedDict([('type', 'pubsub'), ('projectId', args.pubsub_project if args.pubsub_project else properties.VALUES.core.project.GetOrFail()), ('topic', args.pubsub_topic), ('createDisposition', create_disposition)])
        outputs.append(pubsub_config)
    return json.dumps(outputs)