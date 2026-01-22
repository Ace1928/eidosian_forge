from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.util.args import labels_util
def ConfigureOrderedJob(messages, job, args):
    """Add type-specific job configuration to job message."""
    job.labels = labels_util.ParseCreateArgs(args, messages.OrderedJob.LabelsValue)