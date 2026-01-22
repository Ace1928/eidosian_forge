from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.domains import operations
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.command_lib.domains import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def ParseTransferLockState(api_version, transfer_lock_state):
    messages = registrations.GetMessagesModule(api_version)
    if transfer_lock_state is None:
        return None
    return flags.TransferLockEnumMapper(messages).GetEnumForChoice(transfer_lock_state)