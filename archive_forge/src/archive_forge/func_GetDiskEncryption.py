from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import environments as env_util
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetDiskEncryption():
    type_enum = None
    if args.IsSpecified('disk_encryption'):
        instance_message = messages.Instance
        type_enum = arg_utils.ChoiceEnumMapper(arg_name='disk-encryption', message_enum=instance_message.DiskEncryptionValueValuesEnum, include_filter=lambda x: 'UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(args.disk_encryption))
    return type_enum