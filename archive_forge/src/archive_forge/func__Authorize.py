from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import getopt
import textwrap
from gslib import metrics
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.kms_api import KmsApi
from gslib.project_id import PopulateProjectId
from gslib.third_party.kms_apitools.cloudkms_v1_messages import Binding
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import text_util
from gslib.utils.constants import NO_MAX
from gslib.utils.encryption_helper import ValidateCMEK
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
def _Authorize(self):
    self._GatherSubOptions('authorize')
    if not self.kms_key:
        raise CommandException('%s %s requires a key to be specified with -k' % (self.command_name, self.subcommand_name))
    _, newly_authorized = self._AuthorizeProject(self.project_id, self.kms_key)
    if newly_authorized:
        print('Authorized project %s to encrypt and decrypt with key:\n%s' % (self.project_id, self.kms_key))
    else:
        print('Project %s was already authorized to encrypt and decrypt with key:\n%s.' % (self.project_id, self.kms_key))
    return 0