from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.help_provider import CreateHelpText
from gslib.metrics import LogCommandParams
from gslib.project_id import PopulateProjectId
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.text_util import InsistAscii
from gslib.utils import shim_util
def _ListHmacKeys(self, thread_state=None):
    """Lists HMAC keys for a project or service account."""
    if self.args:
        raise CommandException('%s %s received unexpected arguments.\n%s' % (self.command_name, self.action_subcommand, _LIST_SYNOPSIS))
    gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
    response = gsutil_api.ListHmacKeys(self.project_id, self.service_account_email, self.show_all, provider='gs')
    short_list_format = '%s\t%-12s %s'
    if self.long_list:
        for item in response:
            print(_KeyMetadataOutput(item))
            print()
    else:
        for item in response:
            print(short_list_format % (item.accessId, item.state, item.serviceAccountEmail))