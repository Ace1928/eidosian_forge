from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import functools
import itertools
import sys
import textwrap
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.api_lib.ml_engine import versions_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.iam import completers as iam_completers
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.ml_engine import constants
from googlecloudsdk.command_lib.ml_engine import models_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def AddRequestLoggingConfigFlags(parser):
    """Adds flags related to request response logging."""
    group = parser.add_argument_group(help='Configure request response logging.')
    group.add_argument('--bigquery-table-name', type=str, help='Fully qualified name (project_id.dataset_name.table_name) of the BigQuery\ntable where requests and responses are logged.\n')
    group.add_argument('--sampling-percentage', type=float, help='Percentage of requests to be logged, expressed as a number between 0 and 1.\nFor example, set this value to 1 in order to log all requests or to 0.1 to log\n10% of requests.\n')