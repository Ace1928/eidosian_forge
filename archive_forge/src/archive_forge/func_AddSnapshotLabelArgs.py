from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def AddSnapshotLabelArgs(parser):
    labels_util.GetCreateLabelsFlag(extra_message='The label is added to each snapshot created by the schedule.', labels_name='snapshot-labels').AddToParser(parser)