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
class GcloudCrc32cOperation(binary_operations.BinaryBackedOperation):
    """Operation for hashing a file using gcloud-crc32c."""

    def __init__(self, **kwargs):
        super(GcloudCrc32cOperation, self).__init__(binary=BINARY_NAME, **kwargs)

    def _ParseArgsForCommand(self, file_path, offset=0, length=0, **kwargs):
        return ['-o', str(offset), '-l', str(length), file_path]