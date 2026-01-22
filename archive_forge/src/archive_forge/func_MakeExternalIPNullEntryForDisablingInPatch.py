from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.instance_groups import flags
def MakeExternalIPNullEntryForDisablingInPatch(client, interface_name):
    return client.messages.StatefulPolicyPreservedState.ExternalIPsValue.AdditionalProperty(key=interface_name, value=None)