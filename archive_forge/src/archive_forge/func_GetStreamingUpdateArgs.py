from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetStreamingUpdateArgs(required=False):
    """Defines the Streaming Update Args for the Pipeline."""
    streaming_update_args = base.ArgumentGroup(required=required)
    streaming_update_args.AddArgument(base.Argument('--update', required=required, action=arg_parsers.StoreTrueFalseAction, help='Set this to true for streaming update jobs.'))
    streaming_update_args.AddArgument(base.Argument('--transform-name-mappings', required=required, default=None, metavar='TRANSFORM_NAME_MAPPINGS', type=arg_parsers.ArgDict(), action=arg_parsers.UpdateAction, help='Transform name mappings for the streaming update job.'))
    return streaming_update_args