from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.interconnects import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.interconnects import flags
from googlecloudsdk.command_lib.util.args import labels_util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class UpdateLabels(Update):
    """Update a Compute Engine interconnect.

  *{command}* is used to update interconnects. An interconnect represents a
  single specific connection between Google and the customer.
  """

    @classmethod
    def Args(cls, parser):
        _ArgsCommon(cls, parser, support_labels=True)

    def Run(self, args):
        self._DoRun(args, support_labels=True)