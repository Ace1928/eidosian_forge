from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
def GetUpdateRequest(self, client, args, instance_ref, access_config):
    return (client.apitools_client.instances, 'UpdateAccessConfig', client.messages.ComputeInstancesUpdateAccessConfigRequest(instance=instance_ref.instance, networkInterface=args.network_interface, accessConfig=access_config, project=instance_ref.project, zone=instance_ref.zone))