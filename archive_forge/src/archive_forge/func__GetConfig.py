from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.endpoints import services_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.endpoints import arg_parsers
from googlecloudsdk.command_lib.endpoints import common_flags
from googlecloudsdk.core import resources
def _GetConfig(self, service, config_id):
    messages = services_util.GetMessagesModule()
    client = services_util.GetClientInstance()
    request = messages.ServicemanagementServicesConfigsGetRequest(serviceName=service, configId=config_id)
    return client.services_configs.Get(request)