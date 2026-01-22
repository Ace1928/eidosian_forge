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
def _CleanUpUpdateCommand(self, tf, dirs_to_remove, old_cwd):
    """Cleans up temp files etc. from running update command.

    Args:
      tf: Opened TarFile, or None if none currently open.
      dirs_to_remove: List of directories to remove.
      old_cwd: Path to the working directory we should chdir back to. It's
          possible that we've chdir'd to a temp directory that's been deleted,
          which can cause odd behavior (e.g. OSErrors when opening the metrics
          subprocess). If this is not truthy, we won't attempt to chdir back
          to this value.
    """
    if tf:
        tf.close()
    self._EnsureDirsSafeForUpdate(dirs_to_remove)
    for directory in dirs_to_remove:
        try:
            shutil.rmtree(directory)
        except OSError:
            if not system_util.IS_WINDOWS:
                raise
    if old_cwd:
        try:
            os.chdir(old_cwd)
        except OSError:
            pass