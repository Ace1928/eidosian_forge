from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def InterconnectLocationArgument(required=True):
    return compute_flags.ResourceArgument(resource_name='interconnect location', completer=InterconnectLocationsCompleter, plural=False, required=required, global_collection='compute.interconnectLocations')