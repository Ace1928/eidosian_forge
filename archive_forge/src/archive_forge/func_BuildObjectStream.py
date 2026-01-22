from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import sys
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import transfer
from googlecloudsdk.api_lib.dataproc import exceptions as dp_exceptions
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six.moves.urllib.parse
def BuildObjectStream(self, stream, object_ref):
    """Build an apitools Download from a stream and a GCS object reference.

    Note: This will always succeed, but HttpErrors with downloading will be
      raised when the download's methods are called.

    Args:
      stream: An Stream-like object that implements write(<string>) to write
        into.
      object_ref: A proto message of the object to fetch. Only the bucket and
        name need be set.

    Returns:
      The download.
    """
    download = transfer.Download.FromStream(stream, total_size=object_ref.size, auto_transfer=False)
    self._GetObject(object_ref, download=download)
    return download