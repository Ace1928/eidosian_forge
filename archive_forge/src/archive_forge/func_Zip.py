from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import zipfile
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import archive
from googlecloudsdk.core.util import files
from six.moves import urllib
def Zip(self):
    """Creates a zip archive of the specified directory."""
    dst_file = os.path.join(self._tmp_dir.path, self._ARCHIVE_FILE_NAME)
    archive.MakeZipFromDir(dst_file, self._src_dir, self._ZipFileFilter)
    return dst_file