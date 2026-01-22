from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute.instance_templates import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetListImplementation(self, client, args, request_data):
    return lister.MultiScopeLister(client, regional_service=client.apitools_client.regionInstanceTemplates, global_service=client.apitools_client.instanceTemplates, aggregation_service=client.apitools_client.instanceTemplates, instance_view_flag=self._GetInstanceView(args.view, self._getRequest(client.messages, request_data)))