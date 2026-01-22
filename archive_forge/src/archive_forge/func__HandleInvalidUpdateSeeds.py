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
def _HandleInvalidUpdateSeeds(self, diff, version, update_seed):
    """Checks that the update seeds are valid components.

    Args:
      diff: The ComponentSnapshotDiff.
      version: str, The SDK version if in install mode or None if in update
        mode.
      update_seed: [str], A list of component ids to update.

    Raises:
      InvalidComponentError: If any of the given component ids do not exist.

    Returns:
      [str], The update seeds that should be used for the install/update.
    """
    invalid_seeds = diff.InvalidUpdateSeeds(update_seed)
    if not invalid_seeds:
        return update_seed
    if encoding.GetEncodedValue(os.environ, 'CLOUDSDK_REINSTALL_COMPONENTS'):
        return set(update_seed) - invalid_seeds
    ignored = set(_IGNORED_MISSING_COMPONENTS)
    deprecated = invalid_seeds & ignored
    for item in deprecated:
        log.warning('Component [%s] no longer exists.', item)
        additional_msg = _IGNORED_MISSING_COMPONENTS.get(item)
        if additional_msg:
            log.warning(additional_msg)
    invalid_seeds -= ignored
    if invalid_seeds:
        completely_invalid_seeds = invalid_seeds
        update_required_seeds = set()
        if version:
            _, latest_diff = self._GetStateAndDiff(command_path='components.update')
            completely_invalid_seeds = latest_diff.InvalidUpdateSeeds(invalid_seeds)
            update_required_seeds = invalid_seeds - completely_invalid_seeds
        msgs = []
        if completely_invalid_seeds:
            msgs.append('The following components are unknown [{}].'.format(', '.join(completely_invalid_seeds)))
        if update_required_seeds:
            msgs.append('The following components are not available for your current CLI version [{}]. Please run `gcloud components update` to update your Google Cloud CLI.'.format(', '.join(update_required_seeds)))
        raise InvalidComponentError(' '.join(msgs))
    return set(update_seed) - deprecated