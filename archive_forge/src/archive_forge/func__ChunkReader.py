from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
def _ChunkReader(file_, chunk_size=_READ_SIZE):
    while True:
        chunk = file_.read(chunk_size)
        if not chunk:
            break
        yield chunk