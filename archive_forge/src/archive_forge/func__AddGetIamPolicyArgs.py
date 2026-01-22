from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import models
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import endpoint_util
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import models_util
from googlecloudsdk.command_lib.ml_engine import region_util
def _AddGetIamPolicyArgs(parser):
    flags.GetModelResourceArg(positional=True, required=True, verb='to set IAM policy for').AddToParser(parser)
    flags.GetRegionArg(include_global=True).AddToParser(parser)
    base.URI_FLAG.RemoveFromParser(parser)