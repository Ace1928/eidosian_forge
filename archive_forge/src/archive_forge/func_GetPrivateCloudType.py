from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import clusters
from googlecloudsdk.api_lib.vmware import networks
from googlecloudsdk.api_lib.vmware import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.exceptions import Error
def GetPrivateCloudType(self, private_cloud_type):
    type_enum = arg_utils.ChoiceEnumMapper(arg_name='type', default='STANDARD', message_enum=self.messages.PrivateCloud.TypeValueValuesEnum).GetEnumForChoice(arg_utils.EnumNameToChoice(private_cloud_type))
    return type_enum