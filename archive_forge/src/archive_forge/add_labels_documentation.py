from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import labels_doc_helper
from googlecloudsdk.command_lib.compute import labels_flags
from googlecloudsdk.command_lib.compute.snapshots import flags as snapshots_flags
from googlecloudsdk.command_lib.util.args import labels_util
Add labels to Compute Engine snapshots.

  *{command}* adds labels to a Compute Engine snapshot.
  For example, running:

    $ {command} example-snapshot --labels=k0=v0,k1=v1

  will add key-value pairs ``k0''=``v0'' and ``k1''=``v1'' to
  'example-snapshot'.

  Labels can be used to identify the snapshot and to filter them as in

    $ {parent_command} list --filter='labels.k1:value2'

  To list existing labels

    $ {parent_command} describe example-snapshot --format="default(labels)"
  