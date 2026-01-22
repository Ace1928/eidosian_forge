from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
class TargetInstancesCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(TargetInstancesCompleter, self).__init__(collection='compute.targetInstances', list_command='compute target-instances list --uri', **kwargs)