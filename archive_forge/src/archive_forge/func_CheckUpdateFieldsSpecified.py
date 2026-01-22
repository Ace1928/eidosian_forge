from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.beyondcorp.app import util as api_util
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.beyondcorp.app import util as command_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
def CheckUpdateFieldsSpecified(unused_ref, args, patch_request):
    """Check that update command has one of these flags specified."""
    update_args = ['display_name', 'ingress_config', 'config_from_file']
    if any((args.IsSpecified(update_arg) for update_arg in update_args)):
        return patch_request
    raise exceptions.Error('Must specify at least one field to update. Try --help.')