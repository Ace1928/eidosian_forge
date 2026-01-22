from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import random
import re
import string
import time
from typing import Dict, Optional
from apitools.base.py import exceptions as http_exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
from apitools.base.py import util as http_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.functions import exceptions
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.util import archive
from googlecloudsdk.core.util import files as file_utils
import six
from six.moves import http_client
from six.moves import range
def CreateSourcesZipFile(zip_dir: str, source_path: str, ignore_file: Optional[str]=None, enforce_size_limit=False) -> str:
    """Prepare zip file with source of the function to upload.

  Args:
    zip_dir: str, directory in which zip file will be located. Name of the file
      will be `fun.zip`.
    source_path: str, directory containing the sources to be zipped.
    ignore_file: custom ignore_file name. Override .gcloudignore file to
      customize files to be skipped.
    enforce_size_limit: if set, enforces that the unpacked source size is less
      than or equal to 512 MB.

  Returns:
    Path to the zip file.
  Raises:
    FunctionsError
  """
    _ValidateDirectoryExistsOrRaise(source_path)
    if ignore_file and (not os.path.exists(os.path.join(source_path, ignore_file))):
        raise exceptions.IgnoreFileNotFoundError('File {0} referenced by --ignore-file does not exist.'.format(ignore_file))
    if enforce_size_limit:
        _ValidateUnpackedSourceSize(source_path, ignore_file)
    zip_file_name = os.path.join(zip_dir, 'fun.zip')
    try:
        chooser = _GetChooser(source_path, ignore_file)
        predicate = chooser.IsIncluded
        archive.MakeZipFromDir(zip_file_name, source_path, predicate=predicate)
    except ValueError as e:
        raise exceptions.FunctionsError('Error creating a ZIP archive with the source code for directory {0}: {1}'.format(source_path, six.text_type(e)))
    return zip_file_name