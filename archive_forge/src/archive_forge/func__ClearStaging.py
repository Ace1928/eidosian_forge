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
@_RaisesPermissionsError
def _ClearStaging(self, progress_callback=None):
    """Deletes the current staging directory if it exists.

    Args:
      progress_callback: f(float), A function to call with the fraction of
        completeness.
    """
    if os.path.exists(self.__sdk_staging_root):
        file_utils.RmTree(self.__sdk_staging_root)
    if progress_callback:
        progress_callback(1)