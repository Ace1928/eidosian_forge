from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.util.apis import arg_utils
def _IsInternetNeg(self, network_endpoint_type):
    endpoint_type_enum = self.messages.NetworkEndpointGroup.NetworkEndpointTypeValueValuesEnum
    endpoint_type_enum_value = arg_utils.ChoiceToEnum(network_endpoint_type, endpoint_type_enum)
    return endpoint_type_enum_value in {endpoint_type_enum.INTERNET_FQDN_PORT, endpoint_type_enum.INTERNET_IP_PORT}