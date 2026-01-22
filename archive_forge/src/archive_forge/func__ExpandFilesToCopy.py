from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.storage import expansion
from googlecloudsdk.command_lib.storage import paths
from googlecloudsdk.command_lib.storage import storage_parallel
from googlecloudsdk.core import exceptions
def _ExpandFilesToCopy(self, sources):
    """Do initial expansion of all the wildcard arguments.

    Args:
      sources: [paths.Path], The sources (containing optional wildcards) that
        you want to copy.

    Returns:
      ([paths.Path], [paths.Path]), The file and directory paths that the
      initial set of sources expanded to.
    """
    files = set()
    dirs = set()
    for s in sources:
        expander = self._GetExpander(s)
        current_files, current_dirs = expander.ExpandPath(s.path)
        files.update(current_files)
        dirs.update(current_dirs)
    return ([paths.Path(f) for f in sorted(files)], [paths.Path(d) for d in sorted(dirs)])