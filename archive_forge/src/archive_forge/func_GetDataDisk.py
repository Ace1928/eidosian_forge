from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.workbench import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetDataDisk(args, messages):
    """Creates the data disk config for the instance.

  Args:
    args: Argparse object from Command.Run
    messages: Module containing messages definition for the specified API.

  Returns:
    Data disk config for the instance.
  """
    data_disk_message = messages.DataDisk
    data_disk_encryption_enum = None
    data_disk_type_enum = None
    kms_key = None
    if args.IsSpecified('data_disk_type'):
        data_disk_type_enum = arg_utils.ChoiceEnumMapper(arg_name='data-disk-type', message_enum=data_disk_message.DiskTypeValueValuesEnum, include_filter=lambda x: 'UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(args.data_disk_type))
    if args.IsSpecified('data_disk_encryption'):
        data_disk_encryption_enum = arg_utils.ChoiceEnumMapper(arg_name='data-disk-encryption', message_enum=data_disk_message.DiskEncryptionValueValuesEnum, include_filter=lambda x: 'UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(args.data_disk_encryption))
    if args.IsSpecified('data_disk_kms_key'):
        kms_key = args.CONCEPTS.data_disk_kms_key.Parse().RelativeName()
    return data_disk_message(diskType=data_disk_type_enum, diskEncryption=data_disk_encryption_enum, diskSizeGb=args.data_disk_size, kmsKey=kms_key)