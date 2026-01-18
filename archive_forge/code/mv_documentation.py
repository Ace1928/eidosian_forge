from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.commands.cp import CP_AND_MV_SHIM_FLAG_MAP
from gslib.commands.cp import CP_SUB_ARGS
from gslib.commands.cp import ShimTranslatePredefinedAclSubOptForCopy
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import StorageUrlFromString
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageMap
Command entry point for the mv command.