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
def UploadArchive(upload_url, zip_file):
    """Uploads the specified zip file with a PUT request to the provided URL.

  Args:
    upload_url: A string of the URL to send the PUT request to. Required to be a
      signed URL from GCS.
    zip_file: A string of the file path to the zip file to upload.

  Returns:
    A requests.Response object.
  """
    sess = requests.GetSession()
    headers = {'content-type': 'application/zip', 'x-goog-content-length-range': '0,1073741824'}
    with files.BinaryFileReader(zip_file) as data:
        response = sess.put(upload_url, data=data, headers=headers)
    return response