from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import heapq
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import containeranalysis_util as ca_util
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.artifacts import flags
from googlecloudsdk.command_lib.artifacts import format_util
from googlecloudsdk.core import log
def _IncludeMetadata(args):
    default_occ_filter = 'kind="BUILD" OR kind="IMAGE" OR kind="DISCOVERY" OR kind="SBOM_REFERENCE"'
    return args.show_occurrences or (args.occurrence_filter and args.occurrence_filter != default_occ_filter)