from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.storage import expansion
from googlecloudsdk.command_lib.storage import paths
from googlecloudsdk.command_lib.storage import storage_parallel
from googlecloudsdk.core import exceptions
def _GetDirCopyTasks(self, dirs, dest):
    """Get the Tasks to be executed to copy the given directories.

    If dest is dir-like (ending in a slash), all dirs are copied under the
    destination. If it is file-like, at most one directory can be provided and
    it is copied directly to the destination name.

    File copy tasks are generated recursively for the contents of all
    directories.

    Args:
      dirs: [paths.Path], The directories to copy.
      dest: paths.Path, The destination to copy the directories to.

    Returns:
      [storage_parallel.Task], The file copy tasks to execute.
    """
    tasks = []
    for d in dirs:
        item_dest = self._GetDestinationName(d, dest)
        expander = self._GetExpander(d)
        files, sub_dirs = expander.ExpandPath(d.Join('*').path)
        files = [paths.Path(f) for f in sorted(files)]
        sub_dirs = [paths.Path(d) for d in sorted(sub_dirs)]
        tasks.extend(self._GetFileCopyTasks(files, item_dest))
        tasks.extend(self._GetDirCopyTasks(sub_dirs, item_dest))
    return tasks