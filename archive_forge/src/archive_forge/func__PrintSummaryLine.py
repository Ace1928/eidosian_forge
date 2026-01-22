from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import locale
import sys
import six
from gslib.bucket_listing_ref import BucketListingObject
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
from gslib.utils import ls_helper
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import S3_DELETE_MARKER_GUID
from gslib.utils.constants import UTF8
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.text_util import print_to_fd
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils import text_util
def _PrintSummaryLine(self, num_bytes, name):
    size_string = MakeHumanReadable(num_bytes) if self.human_readable else six.text_type(num_bytes)
    text_util.print_to_fd('{size:<11}  {name}'.format(size=size_string, name=six.ensure_text(name)), end=self.line_ending)