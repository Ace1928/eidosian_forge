import functools
import logging
from cinderclient import api_versions
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def _set_encryption_type(volume_client, volume_type, parsed_args):
    body = {}
    for attr in ['provider', 'cipher', 'key_size', 'control_location']:
        info = getattr(parsed_args, 'encryption_' + attr, None)
        if info is not None:
            body[attr] = info
    try:
        volume_client.volume_encryption_types.update(volume_type, body)
    except Exception as e:
        if type(e).__name__ == 'NotFound':
            LOG.warning(_('No existing encryption type found, creating new encryption type for this volume type ...'))
            _create_encryption_type(volume_client, volume_type, parsed_args)