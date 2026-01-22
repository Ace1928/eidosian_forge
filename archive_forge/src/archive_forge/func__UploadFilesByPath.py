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
def _UploadFilesByPath(paths, staging_location):
    """Uploads files after validating and transforming input type."""
    if not staging_location:
        raise NoStagingLocationError()
    counter = collections.Counter(list(map(os.path.basename, paths)))
    duplicates = [name for name, count in six.iteritems(counter) if count > 1]
    if duplicates:
        raise DuplicateEntriesError(duplicates)
    upload_pairs = [(path, os.path.basename(path)) for path in paths]
    return uploads.UploadFiles(upload_pairs, staging_location.bucket_ref, staging_location.name)