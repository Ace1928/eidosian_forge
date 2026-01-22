from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
import signal
import stat
import sys
import tarfile
import tempfile
import textwrap
from six.moves import input
import gslib
from gslib.command import Command
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.metrics import CheckAndMaybePromptForAnalyticsEnabling
from gslib.sig_handling import RegisterSignalHandler
from gslib.utils import system_util
from gslib.utils.boto_util import GetConfigFilePaths
from gslib.utils.boto_util import CERTIFICATE_VALIDATION_ENABLED
from gslib.utils.constants import RELEASE_NOTES_URL
from gslib.utils.text_util import CompareVersions
from gslib.utils.update_util import DisallowUpdateIfDataInGsutilDir
from gslib.utils.update_util import LookUpGsutilVersion
from gslib.utils.update_util import GsutilPubTarball
def _EnsureDirsSafeForUpdate(self, dirs):
    """Raises Exception if any of dirs is known to be unsafe for gsutil update.

    This provides a fail-safe check to ensure we don't try to overwrite
    or delete any important directories. (That shouldn't happen given the
    way we construct tmp dirs, etc., but since the gsutil update cleanup
    uses shutil.rmtree() it's prudent to add extra checks.)

    Args:
      dirs: List of directories to check.

    Raises:
      CommandException: If unsafe directory encountered.
    """
    for d in dirs:
        if not d:
            d = 'null'
        if d.lstrip(os.sep).lower() in self.unsafe_update_dirs:
            raise CommandException('EnsureDirsSafeForUpdate: encountered unsafe directory (%s); aborting update' % d)