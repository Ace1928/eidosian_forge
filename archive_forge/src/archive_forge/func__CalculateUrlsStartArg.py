from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib import metrics
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.text_util import NormalizeStorageClass
from gslib.utils import shim_util
def _CalculateUrlsStartArg(self):
    if not self.args:
        self.RaiseWrongNumberOfArgumentsException()
    if self.args[0].lower() == 'set':
        return 2
    else:
        return 1