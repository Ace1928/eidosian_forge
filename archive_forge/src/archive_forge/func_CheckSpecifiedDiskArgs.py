from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances import utils as instances_utils
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
def CheckSpecifiedDiskArgs(args, support_disks=True, skip_defaults=False, support_kms=False, support_nvdimm=False):
    """Checks if relevant disk arguments have been specified."""
    flags_to_check = ['local_ssd', 'boot_disk_type', 'boot_disk_device_name', 'boot_disk_auto_delete']
    if support_disks:
        flags_to_check.extend(['disk', 'require_csek_key_create'])
    if support_kms:
        flags_to_check.extend(['create_disk', 'boot_disk_kms_key', 'boot_disk_kms_project', 'boot_disk_kms_location', 'boot_disk_kms_keyring'])
    if support_nvdimm:
        flags_to_check.extend(['local_nvdimm'])
    if skip_defaults and (not instance_utils.IsAnySpecified(args, *flags_to_check)):
        return False
    return True