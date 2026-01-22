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
def _CheckIfPathExists(self, path):
    """Checks that the given file path exists."""
    if path and (not os.path.exists(path)):
        raise files.MissingFileError('Path to archive deployment does not exist: {}'.format(path))