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
@contextlib.contextmanager
def _EnsureCaCertsCleanedUpIfObsoleted(self):
    """Context manager that cleans up CA certs file upon exit if it's obsolete.

    When performing an in-place update, we restore the CA certs file to its
    original location if the component containing it needed to be removed. It's
    possible, however, that a newer version of gcloud gets the CA certs file
    from a different path (e.g. if there's a change to certifi or gcloud's
    directory structure at some point in the future). In that case, we want to
    ensure we clean up the old path once it's no longer needed to make requests
    (i.e. just after installing the new components), rather than have it lying
    around forever in the install dir. This context manager takes care of that.

    Yields:
      str, Path to the existing CA certs file.
    """
    ca_certs_path = properties.VALUES.core.custom_ca_certs_file.Get() or certifi.where()
    initial_state = self._GetInstallState()
    initial_components = initial_state.InstalledComponents()
    initial_paths = set(itertools.chain.from_iterable((manifest.InstalledPaths() for manifest in initial_components.values())))
    try:
        yield ca_certs_path
    finally:
        current_state = self._GetInstallState()
        current_components = current_state.InstalledComponents()
        current_paths = set(itertools.chain.from_iterable((manifest.InstalledPaths() for manifest in current_components.values())))
        removed_paths = initial_paths - current_paths
        try:
            relative_ca_certs_path = pathlib.Path(ca_certs_path).resolve().relative_to(pathlib.Path(self.__sdk_root).resolve()).as_posix()
        except ValueError:
            pass
        else:
            if relative_ca_certs_path in removed_paths:
                os.remove(ca_certs_path)
                dir_path = os.path.dirname(os.path.normpath(relative_ca_certs_path))
                full_dir_path = os.path.join(self.__sdk_root, dir_path)
                while dir_path and (not os.listdir(full_dir_path)):
                    os.rmdir(full_dir_path)
                    dir_path = os.path.dirname(dir_path)
                    full_dir_path = os.path.join(self.__sdk_root, dir_path)