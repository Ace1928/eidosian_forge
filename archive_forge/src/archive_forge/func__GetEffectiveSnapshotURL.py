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
def _GetEffectiveSnapshotURL(self, version=None):
    """Get the snapshot URL we shoould download based on any override versions.

    This starts with the configured URL (or comma separated URL list) and
    potentially modifies it based on the version.  If a version is specified,
    it is converted to the fixed version specific snapshot.  If the SDK is set
    to use a fixed version, that is then used.  If neither, the original URL
    is used.

    Args:
      version: str, The Cloud SDK version to get the snapshot for.

    Raises:
      MismatchedFixedVersionsError: If you manually specify a version and you
        are fixed to a different version.

    Returns:
      str, The modified snapshot URL.
    """
    url = self.__base_url
    if version:
        if self.__fixed_version and self.__fixed_version != version:
            raise MismatchedFixedVersionsError('You have configured your Google Cloud CLI installation\nto be fixed to version [{0}] but are attempting to install components at\nversion [{1}].  To clear your fixed version setting, run:\n    $ gcloud config unset component_manager/fixed_sdk_version'.format(self.__fixed_version, version))
    elif self.__fixed_version:
        if self.__warn:
            log.warning('You have configured your Google Cloud CLI installation to be fixed to version [{0}].'.format(self.__fixed_version))
        version = self.__fixed_version
    if version:
        urls = url.split(',')
        urls[0] = os.path.dirname(urls[0]) + '/' + UpdateManager.VERSIONED_SNAPSHOT_FORMAT.format(version)
        url = ','.join(urls)
    repos = properties.VALUES.component_manager.additional_repositories.Get()
    if repos:
        if self.__warn:
            for repo in repos.split(','):
                log.warning('You are using additional component repository: [%s]', repo)
        url = ','.join([url, repos])
    return url