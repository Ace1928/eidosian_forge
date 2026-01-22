from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import compileall
import errno
import logging
import os
import posixpath
import re
import shutil
import sys
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
import six
def MarkInstalled(self, snapshot, files):
    """Marks this component as installed with the given snapshot and files.

    This saves the ComponentSnapshot and writes the installed files to a
    manifest so they can be removed later.

    Args:
      snapshot: snapshots.ComponentSnapshot, The snapshot that was the source
        of the install.
      files: list of str, The files that were created by the installation.
    """
    with file_utils.FileWriter(self.manifest_file) as fp:
        for f in _NormalizeFileList(files):
            fp.write(f + '\n')
    snapshot.WriteToFile(self.snapshot_file, component_id=self.id)