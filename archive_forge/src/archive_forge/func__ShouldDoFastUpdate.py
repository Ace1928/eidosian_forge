from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import hashlib
import itertools
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import release_notes
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.updater import update_check
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
from six.moves import map  # pylint: disable=redefined-builtin
def _ShouldDoFastUpdate(self, allow_no_backup=False, fast_mode_impossible=False, has_components_to_remove=False):
    """Determine whether we should do an in-place fast update or make a backup.

    This method also ensures the CWD is valid for the mode we are going to use.

    Args:
      allow_no_backup: bool, True if we want to allow the updater to run
        without creating a backup.  This lets us be in the root directory of the
        SDK and still do an update.  It is more fragile if there is a failure,
        so we only do it if necessary.
      fast_mode_impossible: bool, True if we can't do a fast update for this
        particular operation (overrides forced fast mode).
      has_components_to_remove: bool, Whether the update operation involves
        removing any components. Affects whether we can perform an in-place
        update from inside the root dir.

    Returns:
      bool, True if allow_no_backup was True and we are under the SDK root (so
        we should do a no backup update).

    Raises:
      InvalidCWDError: If the command is run from a directory within the SDK
        root.
    """
    force_fast = properties.VALUES.experimental.fast_component_update.GetBool()
    if fast_mode_impossible:
        force_fast = False
    cwd = None
    try:
        cwd = os.path.realpath(file_utils.GetCWD())
    except OSError:
        log.debug('Could not determine CWD, assuming detached directory not under SDK root.')
    if not (cwd and file_utils.IsDirAncestorOf(self.__sdk_root, cwd)):
        return force_fast
    if (allow_no_backup or force_fast) and (not has_components_to_remove):
        return True
    raise InvalidCWDError('Your current working directory is inside the Google Cloud CLI install root: {root}.  In order to perform this update, run the command from outside of this directory.'.format(root=self.__sdk_root))