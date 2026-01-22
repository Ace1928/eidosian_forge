from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import os
import subprocess
import sys
import threading
from . import comm
import ruamel.yaml as yaml
from six.moves import input
class GeneratedFile(object):
    """Wraps the name and contents of a generated file."""

    def __init__(self, filename, contents):
        """Constructor.

    Args:
      filename: (str) Unix style file path relative to the target source
        directory.
      contents: (str) File contents.
    """
        self.filename = filename
        self.contents = contents

    def WriteTo(self, dest_dir, notify):
        """Write the file to the destination directory.

    Args:
      dest_dir: (str) Destination directory.
      notify: (callable(str)) Function to notify the user.

    Returns:
      (str or None) The full normalized path name of the destination file,
      None if it wasn't generated because it already exists.
    """
        path = _NormalizePath(dest_dir, self.filename)
        if not os.path.exists(path):
            notify(WRITING_FILE_MESSAGE.format(self.filename, dest_dir))
            with open(path, 'w') as f:
                f.write(self.contents)
            return path
        else:
            notify(FILE_EXISTS_MESSAGE.format(self.filename))
        return None