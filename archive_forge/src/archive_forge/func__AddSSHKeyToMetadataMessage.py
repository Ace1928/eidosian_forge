from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import collections
import datetime
import json
import os
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
from googlecloudsdk.core.util.files import FileReader
from googlecloudsdk.core.util.files import FileWriter
import six
def _AddSSHKeyToMetadataMessage(message_classes, user, public_key, metadata, expiration=None, legacy=False):
    """Adds the public key material to the metadata if it's not already there.

  Args:
    message_classes: An object containing API message classes.
    user: The username for the SSH key.
    public_key: The SSH public key to add to the metadata.
    metadata: The existing metadata.
    expiration: If provided, a datetime after which the key is no longer valid.
    legacy: If true, store the key in the legacy "sshKeys" metadata entry.

  Returns:
    An updated metadata API message.
  """
    if expiration is None:
        entry = '{user}:{public_key}'.format(user=user, public_key=public_key.ToEntry(include_comment=True))
    else:
        expire_on = times.FormatDateTime(expiration, '%Y-%m-%dT%H:%M:%S+0000', times.UTC)
        entry = '{user}:{public_key} google-ssh {jsondict}'.format(user=user, public_key=public_key.ToEntry(include_comment=False), jsondict=json.dumps(collections.OrderedDict([('userName', user), ('expireOn', expire_on)])).replace(' ', ''))
    ssh_keys, ssh_legacy_keys = _GetSSHKeysFromMetadata(metadata)
    all_ssh_keys = ssh_keys + ssh_legacy_keys
    log.debug('Current SSH keys in project: {0}'.format(all_ssh_keys))
    if entry in all_ssh_keys:
        return metadata
    if legacy:
        metadata_key = constants.SSH_KEYS_LEGACY_METADATA_KEY
        updated_ssh_keys = ssh_legacy_keys
    else:
        metadata_key = constants.SSH_KEYS_METADATA_KEY
        updated_ssh_keys = ssh_keys
    updated_ssh_keys.append(entry)
    return metadata_utils.ConstructMetadataMessage(message_classes=message_classes, metadata={metadata_key: _PrepareSSHKeysValue(updated_ssh_keys)}, existing_metadata=metadata)