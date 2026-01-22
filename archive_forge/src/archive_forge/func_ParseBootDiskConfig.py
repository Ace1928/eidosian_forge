from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as sdk_core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def ParseBootDiskConfig(boot_disk_args) -> GetMessagesModule('v2alpha1').BootDiskConfig:
    """Parses configurations for boot disk. Boot disk is only in v2alpha1 API.

  Parsing boot disk configuration if --boot-disk flag is set.

  Args:
    boot_disk_args: args for --boot-disk flag.

  Returns:
    Return GetMessagesModule().BootDiskConfig object with parsed configurations.

  Raises:
    BootDiskConfigurationError: if confidential compute is enable
      but kms-key is not provided.
    BootDiskConfigurationError: if invalid argument name is provided.
  """
    tpu_messages = GetMessagesModule('v2alpha1')
    kms_key_arg_name = 'kms-key'
    confidential_compute_arg_name = 'confidential-compute'
    for arg_name in boot_disk_args.keys():
        if arg_name not in [kms_key_arg_name, confidential_compute_arg_name]:
            raise BootDiskConfigurationError('--boot-disk only supports arguments: %s and %s' % (confidential_compute_arg_name, kms_key_arg_name))
    enable_confidential_compute = boot_disk_args.get(confidential_compute_arg_name, 'False').lower() == 'true'
    kms_key = boot_disk_args.get(kms_key_arg_name, None)
    if enable_confidential_compute and kms_key is None:
        raise BootDiskConfigurationError('argument --boot-disk: with confidential-compute=%s requires kms-key; received: %s' % (enable_confidential_compute, kms_key))
    customer_encryption_key = tpu_messages.CustomerEncryptionKey(kmsKeyName=kms_key)
    return tpu_messages.BootDiskConfig(customerEncryptionKey=customer_encryption_key, enableConfidentialCompute=enable_confidential_compute)