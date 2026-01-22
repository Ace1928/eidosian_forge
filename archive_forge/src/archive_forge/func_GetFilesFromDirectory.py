from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import fnmatch
import glob
import importlib.util
import os
import pkgutil
import sys
import types
from googlecloudsdk.core.util import files
def GetFilesFromDirectory(path_dir, filter_pattern='*.*'):
    """Get files from a given directory that match a pattern.

  Args:
    path_dir: str, filesystem path to directory
    filter_pattern: str, pattern to filter files to retrieve.

  Returns:
    List of filtered files from a directory.

  Raises:
    IOError: if resource is not found under given path.
  """
    if os.path.isdir(path_dir):
        return glob.glob(f'{path_dir}/{filter_pattern}')
    else:
        importer = pkgutil.get_importer(path_dir)
        if not hasattr(importer, 'get_data'):
            raise IOError('Path not found {0}'.format(path_dir))
        filtered_files = []
        for file_path in importer._files:
            if not file_path.startswith(importer.prefix):
                continue
            file_path_parts = file_path[len(importer.prefix):].split(os.sep)
            if len(file_path_parts) != 1:
                continue
            if fnmatch.fnmatch(file_path_parts[0], f'{filter_pattern}'):
                filtered_files.append(os.path.join(path_dir, file_path_parts[0]))
        return filtered_files