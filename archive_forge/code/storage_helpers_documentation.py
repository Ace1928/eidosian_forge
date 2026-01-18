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
Read from this stream into a writable.

    Reads at most n bytes, or until it sees there is not a next object in the
    series. This will block for the duration of each object's download,
    and possibly indefinitely if new objects are being added to the channel
    frequently enough.

    Args:
      writable: The stream-like object that implements write(<string>) to
          write into.
      n: A maximum number of bytes to read. Defaults to sys.maxsize
        (usually ~4 GB).

    Raises:
      ValueError: If the stream is closed or objects in the series are
        detected to shrink.

    Returns:
      The number of bytes read.
    