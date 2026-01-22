from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import textwrap
from typing import Mapping
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def LastUpdatedMessage(record):
    if record.status is None:
        return 'Unknown update information'
    modifier = record.last_modifier or '?'
    last_transition_time = '?'
    for condition in record.status.conditions:
        if condition.type == 'Ready' and condition.lastTransitionTime:
            last_transition_time = condition.lastTransitionTime
    return 'Last updated on {} by {}'.format(last_transition_time, modifier)