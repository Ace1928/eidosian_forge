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
def ValidateZipFilePath(self, zip_path):
    """Checks that the zip file path exists and the file is a zip archvie."""
    self._CheckIfPathExists(zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise errors.BundleFileNotValidError(zip_path)