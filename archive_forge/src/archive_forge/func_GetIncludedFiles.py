from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import enum
from googlecloudsdk.command_lib.util import glob
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
from six.moves import map  # pylint: disable=redefined-builtin
def GetIncludedFiles(self, upload_directory, include_dirs=True):
    """Yields the files in the given directory that this FileChooser includes.

    Args:
      upload_directory: str, the path of the directory to upload.
      include_dirs: bool, whether to include directories

    Yields:
      str, the files and directories that should be uploaded.
    Raises:
      SymlinkLoopError: if there is a symlink referring to its own containing
      dir or itself.
    """
    for dirpath, orig_dirnames, filenames in os.walk(six.ensure_str(upload_directory), followlinks=True):
        dirpath = encoding.Decode(dirpath)
        dirnames = [encoding.Decode(dirname) for dirname in orig_dirnames]
        filenames = [encoding.Decode(filename) for filename in filenames]
        if dirpath == upload_directory:
            relpath = ''
        else:
            relpath = os.path.relpath(dirpath, upload_directory)
        for filename in filenames:
            file_relpath = os.path.join(relpath, filename)
            self._RaiseOnSymlinkLoop(os.path.join(dirpath, filename))
            if self.IsIncluded(file_relpath):
                yield file_relpath
        for dirname in dirnames:
            file_relpath = os.path.join(relpath, dirname)
            full_path = os.path.join(dirpath, dirname)
            if self.IsIncluded(file_relpath, is_dir=True):
                self._RaiseOnSymlinkLoop(full_path)
                if include_dirs:
                    yield file_relpath
            else:
                orig_dirnames.remove(dirname)