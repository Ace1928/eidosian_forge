from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetIndexUpdateMethod(required=False):
    return base.Argument('--index-update-method', required=required, type=str, help='The update method to use with this index. Choose `stream_update` or\n`batch_update`. If not set, batch update will be used by default.\n\n`batch_update`: can update index with `gcloud ai indexes update` using\ndatapoints files on Cloud Storage.\n\n`stream update`: can update datapoints with `upsert-datapoints` and\n`delete-datapoints` and will be applied nearly real-time.\n')