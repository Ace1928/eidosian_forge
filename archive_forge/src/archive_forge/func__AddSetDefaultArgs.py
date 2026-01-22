from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import versions_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import endpoint_util
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import region_util
from googlecloudsdk.command_lib.ml_engine import versions_util
def _AddSetDefaultArgs(parser):
    flags.GetModelName(positional=False, required=True).AddToParser(parser)
    flags.GetRegionArg(include_global=True).AddToParser(parser)
    flags.VERSION_NAME.AddToParser(parser)