from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import jobs_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
def _AddUpdateArgs(parser):
    """Get arguments for the `ai-platform jobs update` command."""
    flags.JOB_NAME.AddToParser(parser)
    labels_util.AddUpdateLabelsFlags(parser)