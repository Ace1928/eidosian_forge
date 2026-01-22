from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import compileall
import errno
import logging
import os
import posixpath
import re
import shutil
import sys
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
import six
@_RaisesPermissionsError
def CloneToStaging(self, progress_callback=None):
    """Clones this state to the temporary staging area.

    This is used for making temporary copies of the entire Cloud SDK
    installation when doing updates.  The entire installation is cloned, but
    doing so removes any backups and trash from this state before doing the
    copy.

    Args:
      progress_callback: f(float), A function to call with the fraction of
        completeness.

    Returns:
      An InstallationState object for the cloned install.
    """
    self._CreateStateDir()
    rm_staging_cb, rm_backup_cb, rm_trash_cb, copy_cb = console_io.SplitProgressBar(progress_callback, [1, 1, 1, 7])
    self._ClearStaging(progress_callback=rm_staging_cb)
    self.ClearBackup(progress_callback=rm_backup_cb)
    self.ClearTrash(progress_callback=rm_trash_cb)

    class Counter(object):

        def __init__(self, progress_callback, total):
            self.count = 0
            self.progress_callback = progress_callback
            self.total = total

        def Tick(self, *unused_args):
            self.count += 1
            self.progress_callback(self.count / self.total)
            return []
    if progress_callback:
        dirs = set()
        for _, manifest in six.iteritems(self.InstalledComponents()):
            dirs.update(manifest.InstalledDirectories())
        total_dirs = len(dirs) + 2
        ticker = Counter(copy_cb, total_dirs).Tick if total_dirs else None
    else:
        ticker = None
    shutil.copytree(self.__sdk_root, self.__sdk_staging_root, symlinks=True, ignore=ticker)
    staging_state = InstallationState(self.__sdk_staging_root)
    staging_state._CreateStateDir()
    return staging_state