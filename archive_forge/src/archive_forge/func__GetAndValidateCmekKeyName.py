from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import instance_prop_reducers as reducers
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def _GetAndValidateCmekKeyName(args, is_primary):
    """Parses the CMEK resource arg, makes sure the key format was correct."""
    kms_ref = args.CONCEPTS.kms_key.Parse()
    if kms_ref:
        if is_primary:
            _ShowCmekPrompt()
        return kms_ref.RelativeName()
    else:
        for keyword in ['disk-encryption-key', 'disk-encryption-key-keyring', 'disk-encryption-key-location', 'disk-encryption-key-project']:
            if getattr(args, keyword.replace('-', '_'), None):
                raise exceptions.InvalidArgumentException('--disk-encryption-key', 'not fully specified.')