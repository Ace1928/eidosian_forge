from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetWorkerRegionArgs(required=False):
    """Defines the Streaming Update Args for the Pipeline."""
    worker_region_args = base.ArgumentGroup(required=required, mutex=True)
    worker_region_args.AddArgument(base.Argument('--worker-region', required=required, default=None, help='Default Compute Engine region in which worker processing will occur.'))
    worker_region_args.AddArgument(base.Argument('--worker-zone', required=required, default=None, help='Default Compute Engine zone in which worker processing will occur.'))
    return worker_region_args