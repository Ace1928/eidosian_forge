from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.kuberun import kubernetesobject
from googlecloudsdk.command_lib.kuberun import kubernetes_consts
from googlecloudsdk.core.console import console_attr
def FormatLastUpdated(record):
    modifier = record.last_modifier or '?'
    last_transition_time = '?'
    for condition in record.status.conditions:
        if condition.type == kubernetes_consts.VAL_READY and condition.lastTransitionTime:
            last_transition_time = condition.lastTransitionTime
    return 'Last updated on {} by {}'.format(last_transition_time, modifier)