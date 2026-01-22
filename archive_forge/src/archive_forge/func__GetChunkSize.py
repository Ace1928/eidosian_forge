from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import mimetypes
import os
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from apitools.base.py import transfer
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions as core_exc
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import scaled_integer
import six
def _GetChunkSize(self):
    """Returns the property defined chunksize corrected for server granularity.

    Chunk size for GCS must be a multiple of 256 KiB. This functions rounds up
    the property defined chunk size to the nearest chunk size interval.
    """
    gcs_chunk_granularity = 256 * 1024
    chunksize = scaled_integer.ParseInteger(properties.VALUES.storage.upload_chunk_size.Get())
    if chunksize == 0:
        chunksize = None
    elif chunksize % gcs_chunk_granularity != 0:
        chunksize += gcs_chunk_granularity - chunksize % gcs_chunk_granularity
    return chunksize