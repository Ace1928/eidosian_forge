from the command line arguments and returns a list of URLs to be given to the
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import io
import os
import sys
import textwrap
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.ml_engine import uploads
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
from six.moves import map
from setuptools import setup, find_packages
def UploadPythonPackages(packages=(), package_path=None, staging_location=None):
    """Uploads Python packages (if necessary), building them as-specified.

  An AI Platform job needs one or more Python packages to run. These Python
  packages can be specified in one of three ways:

    1. As a path to a local, pre-built Python package file.
    2. As a path to a Cloud Storage-hosted, pre-built Python package file (paths
       beginning with 'gs://').
    3. As a local Python source tree (the `--package-path` flag).

  In case 1, we upload the local files to Cloud Storage[1] and provide their
  paths. These can then be given to the AI Platform API, which can fetch
  these files.

  In case 2, we don't need to do anything. We can just send these paths directly
  to the AI Platform API.

  In case 3, we perform a build using setuptools[2], and upload the resulting
  artifacts to Cloud Storage[1]. The paths to these artifacts can be given to
  the AI Platform API. See the `BuildPackages` method.

  These methods of specifying Python packages may be combined.


  [1] Uploads are to a specially-prefixed location in a user-provided Cloud
  Storage staging bucket. If the user provides bucket `gs://my-bucket/`, a file
  `package.tar.gz` is uploaded to
  `gs://my-bucket/<job name>/<checksum>/package.tar.gz`.

  [2] setuptools must be installed on the local user system.

  Args:
    packages: list of str. Path to extra tar.gz packages to upload, if any. If
      empty, a package_path must be provided.
    package_path: str. Relative path to source directory to be built, if any. If
      omitted, one or more packages must be provided.
    staging_location: storage_util.ObjectReference. Cloud Storage prefix to
      which archives are uploaded. Not necessary if only remote packages are
      given.

  Returns:
    list of str. Fully qualified Cloud Storage URLs (`gs://..`) from uploaded
      packages.

  Raises:
    ValueError: If packages is empty, and building package_path produces no
      tar archives.
    SetuptoolsFailedError: If the setup.py file fails to successfully build.
    MissingInitError: If the package doesn't contain an `__init__.py` file.
    DuplicateEntriesError: If multiple files with the same name were provided.
    ArgumentError: if no packages were found in the given path or no
      staging_location was but uploads were required.
  """
    remote_paths = []
    local_paths = []
    for package in packages:
        if storage_util.ObjectReference.IsStorageUrl(package):
            remote_paths.append(package)
        else:
            local_paths.append(package)
    if package_path:
        package_root = os.path.dirname(os.path.abspath(package_path))
        with _TempDirOrBackup(package_root) as working_dir:
            local_paths.extend(BuildPackages(package_path, os.path.join(working_dir, 'output')))
            remote_paths.extend(_UploadFilesByPath(local_paths, staging_location))
    elif local_paths:
        remote_paths.extend(_UploadFilesByPath(local_paths, staging_location))
    return remote_paths