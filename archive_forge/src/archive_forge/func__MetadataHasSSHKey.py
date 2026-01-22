from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import sys
import threading
import time
from apitools.base.py import encoding_helper
from apitools.base.py.exceptions import HttpConflictError
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import exceptions as tpu_exceptions
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import util as tpu_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util.files import FileWriter
import six
def _MetadataHasSSHKey(metadata, public_key):
    """Returns true if the metadata has the SSH key.

  Args:
    metadata: Project metadata.
    public_key: The SSH key.

  Returns:
    True if present, False if not present.
  """
    if not (metadata and metadata.items):
        return False
    matching_values = [item.value for item in metadata.items if item.key == SSH_KEYS_METADATA_KEY]
    if not matching_values:
        return False
    return public_key in matching_values[0]