from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.core.util import times
import six
def MakeGuestAccelerators(messages, accelerator_configs):
    """Constructs the repeated accelerator message objects."""
    if accelerator_configs is None:
        return []
    accelerators = []
    for a in accelerator_configs:
        m = messages.AcceleratorConfig(acceleratorCount=a['count'], acceleratorType=a['type'])
        accelerators.append(m)
    return accelerators