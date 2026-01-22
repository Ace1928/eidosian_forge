from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def MaybeGetKmsKey(args, messages, current_value, boot_disk_prefix=False, instance_prefix=False):
    """Gets the Cloud KMS CryptoKey reference from command arguments.

  Args:
    args: Namespaced command line arguments.
    messages: Compute API messages module.
    current_value: Current CustomerEncryptionKey value.
    boot_disk_prefix: If the key flags have the 'boot-disk' prefix.
    instance_prefix: If the key flags have the 'instance' prefix.

  Returns:
    CustomerEncryptionKey message with the KMS key populated if args has a key.
  Raises:
    ConflictingArgumentsException if an encryption key is already populated.
  """
    if boot_disk_prefix:
        key_arg = args.CONCEPTS.boot_disk_kms_key
        flag = '--boot-disk-kms-key'
    elif instance_prefix:
        key_arg = args.CONCEPTS.instance_kms_key
        flag = '--instance-kms-key'
    else:
        key_arg = args.CONCEPTS.kms_key
        flag = '--kms-key'
    key = key_arg.Parse()
    if flag in _GetSpecifiedKmsArgs(args) and (not key):
        raise calliope_exceptions.InvalidArgumentException(flag, 'KMS cryptokey resource was not fully specified.')
    if key:
        if current_value:
            raise calliope_exceptions.ConflictingArgumentsException('--csek-key-file', *_GetSpecifiedKmsArgs(args))
        return messages.CustomerEncryptionKey(kmsKeyName=key.RelativeName())
    return current_value