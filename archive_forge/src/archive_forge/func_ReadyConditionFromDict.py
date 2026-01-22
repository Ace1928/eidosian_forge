from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.kuberun import kubernetesobject
from googlecloudsdk.command_lib.kuberun import kubernetes_consts
from googlecloudsdk.core.console import console_attr
def ReadyConditionFromDict(record):
    ready_cond = [x for x in record.get(kubernetes_consts.FIELD_STATUS, {}).get(kubernetes_consts.FIELD_CONDITIONS, []) if x[kubernetes_consts.FIELD_TYPE] == kubernetes_consts.VAL_READY]
    if ready_cond:
        return ready_cond[0]
    else:
        return None