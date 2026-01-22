from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_templates import flags
@staticmethod
def GetRequestMessage(client, ref):
    if ref.Collection() == 'compute.instanceTemplates':
        return client.messages.ComputeInstanceTemplatesDeleteRequest
    else:
        return client.messages.ComputeRegionInstanceTemplatesDeleteRequest