from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import transfer
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
Upload a file to the GCS results bucket using the storage API.

    Args:
      path: str, the absolute or relative path of the file to upload. File may
        be in located in GCS or the local filesystem.
      mimetype: str, the MIME type (aka Content-Type) that should be applied to
        files being copied from a non-GCS source to GCS. MIME types for GCS->GCS
        file uploads are not modified.
      destination_object: str, the destination object path in GCS to upload to,
        if it's different than the base name of the path argument.

    Raises:
      BadFileException if the file upload is not successful.
    