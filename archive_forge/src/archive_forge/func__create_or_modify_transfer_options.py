from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.command_lib.transfer import jobs_flag_util
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
def _create_or_modify_transfer_options(transfer_spec, args, messages):
    """Creates or modifies TransferOptions object based on args."""
    if not (getattr(args, 'overwrite_when', None) or getattr(args, 'delete_from', None) or getattr(args, 'preserve_metadata', None) or getattr(args, 'custom_storage_class', None)):
        return
    if not transfer_spec.transferOptions:
        transfer_spec.transferOptions = messages.TransferOptions()
    overwrite_when_argument = getattr(args, 'overwrite_when', None)
    if overwrite_when_argument:
        transfer_spec.transferOptions.overwriteWhen = getattr(messages.TransferOptions.OverwriteWhenValueValuesEnum, overwrite_when_argument.upper())
    if getattr(args, 'delete_from', None):
        delete_option = jobs_flag_util.DeleteOption(args.delete_from)
        if delete_option is jobs_flag_util.DeleteOption.SOURCE_AFTER_TRANSFER:
            transfer_spec.transferOptions.deleteObjectsFromSourceAfterTransfer = True
        elif delete_option is jobs_flag_util.DeleteOption.DESTINATION_IF_UNIQUE:
            transfer_spec.transferOptions.deleteObjectsUniqueInSink = True
    metadata_options = messages.MetadataOptions()
    if getattr(args, 'preserve_metadata', None):
        for field_value in args.preserve_metadata:
            field_key = jobs_flag_util.PreserveMetadataField(field_value)
            if field_key == jobs_flag_util.PreserveMetadataField.ACL:
                metadata_options.acl = messages.MetadataOptions.AclValueValuesEnum.ACL_PRESERVE
            elif field_key == jobs_flag_util.PreserveMetadataField.GID:
                metadata_options.gid = messages.MetadataOptions.GidValueValuesEnum.GID_NUMBER
            elif field_key == jobs_flag_util.PreserveMetadataField.UID:
                metadata_options.uid = messages.MetadataOptions.UidValueValuesEnum.UID_NUMBER
            elif field_key == jobs_flag_util.PreserveMetadataField.KMS_KEY:
                metadata_options.kmsKey = messages.MetadataOptions.KmsKeyValueValuesEnum.KMS_KEY_PRESERVE
            elif field_key == jobs_flag_util.PreserveMetadataField.MODE:
                metadata_options.mode = messages.MetadataOptions.ModeValueValuesEnum.MODE_PRESERVE
            elif field_key == jobs_flag_util.PreserveMetadataField.STORAGE_CLASS:
                metadata_options.storageClass = messages.MetadataOptions.StorageClassValueValuesEnum.STORAGE_CLASS_PRESERVE
            elif field_key == jobs_flag_util.PreserveMetadataField.SYMLINK:
                metadata_options.symlink = messages.MetadataOptions.SymlinkValueValuesEnum.SYMLINK_PRESERVE
            elif field_key == jobs_flag_util.PreserveMetadataField.TEMPORARY_HOLD:
                metadata_options.temporaryHold = messages.MetadataOptions.TemporaryHoldValueValuesEnum.TEMPORARY_HOLD_PRESERVE
            elif field_key == jobs_flag_util.PreserveMetadataField.TIME_CREATED:
                metadata_options.timeCreated = messages.MetadataOptions.TimeCreatedValueValuesEnum.TIME_CREATED_PRESERVE_AS_CUSTOM_TIME
    if getattr(args, 'custom_storage_class', None):
        metadata_options.storageClass = getattr(messages.MetadataOptions.StorageClassValueValuesEnum, 'STORAGE_CLASS_' + args.custom_storage_class.upper())
    if metadata_options != messages.MetadataOptions():
        transfer_spec.transferOptions.metadataOptions = metadata_options