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
def AddSSHKeyIfNeeded(project, tpu_helper, node, tpu_name, zone, public_key):
    """Verifies that instance has SSH key, and adds it in case it doesn't."""
    if _MetadataHasSSHKey(project.commonInstanceMetadata, public_key):
        log.status.Print('SSH key found in project metadata; not updating instance.')
        return
    node_dict = encoding_helper.MessageToDict(node)
    ssh_keys = ''
    if 'metadata' in node_dict and SSH_KEYS_METADATA_KEY in node_dict['metadata']:
        ssh_keys = node_dict['metadata'][SSH_KEYS_METADATA_KEY]
    if public_key in ssh_keys:
        log.debug('SSH key found in instance metadata; not updating instance.')
        return
    ssh_keys += '\n' + public_key
    node_for_update = tpu_helper.messages.Node(metadata=tpu_helper.UpdateMetadataKey(metadata=node.metadata, key=SSH_KEYS_METADATA_KEY, value=ssh_keys))
    try:
        tpu_helper.UpdateNode(tpu_name, zone, node_for_update, 'metadata', 'Propagating SSH public key to all TPU workers')
    except HttpConflictError:
        pass