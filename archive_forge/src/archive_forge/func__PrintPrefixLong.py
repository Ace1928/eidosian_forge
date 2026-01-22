from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import six
from gslib.cloud_api import NotFoundException
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import S3_DELETE_MARKER_GUID
from gslib.utils.constants import UTF8
from gslib.utils.ls_helper import ENCRYPTED_FIELDS
from gslib.utils.ls_helper import LsHelper
from gslib.utils.ls_helper import PrintFullInfoAboutObject
from gslib.utils.ls_helper import UNENCRYPTED_FULL_LISTING_FIELDS
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.text_util import InsistAscii
from gslib.utils import text_util
from gslib.utils.translation_helper import AclTranslation
from gslib.utils.translation_helper import LabelTranslation
from gslib.utils.unit_util import MakeHumanReadable
def _PrintPrefixLong(blr):
    text_util.print_to_fd('%-33s%s' % ('', six.ensure_text(blr.url_string)))