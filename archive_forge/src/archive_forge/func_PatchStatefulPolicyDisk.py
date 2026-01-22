from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.instance_groups import flags
def PatchStatefulPolicyDisk(preserved_state, patch):
    """Patch the preserved state proto."""
    if patch.value.autoDelete:
        preserved_state.value.autoDelete = patch.value.autoDelete