from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import struct
import textwrap
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.util import crc32c
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _get_hash_check_warning_base():
    """CRC32C warnings share this text."""
    google_crc32c_install_step = get_google_crc32c_install_command()
    gcloud_crc32c_install_step = 'gcloud components install gcloud-crc32c'
    return textwrap.dedent('      This copy {{}} since fast hash calculation tools\n      are not installed. You can change this by running:\n      \t$ {crc32c_step}\n      You can also modify the "storage/check_hashes" config setting.'.format(crc32c_step=google_crc32c_install_step if google_crc32c_install_step else gcloud_crc32c_install_step))