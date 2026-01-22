from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import os
import shutil
import time
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import parallel
from googlecloudsdk.core.util import text
import six
def _FileDiff(file):
    """Diffs a file in new_dir and old_dir."""
    new_contents, new_binary = GetFileContents(os.path.join(new_dir, file))
    if not new_binary:
        diff.Validate(file, new_contents)
    if file in old_files:
        old_contents, old_binary = GetFileContents(os.path.join(old_dir, file))
        if old_binary == new_binary and old_contents == new_contents:
            return
        return ('edit', file, old_contents, new_contents)
    else:
        return ('add', file, None, new_contents)