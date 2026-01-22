from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import hashlib
import json
import os
import re
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import scaled_integer
def _write_tracker_file(tracker_file_path, data):
    """Creates a tracker file, storing the input data."""
    log.debug('Writing tracker file to {}.'.format(tracker_file_path))
    try:
        file_descriptor = os.open(tracker_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 384)
        with os.fdopen(file_descriptor, 'w') as write_stream:
            write_stream.write(data)
    except OSError as e:
        raise _get_unwritable_tracker_file_error(e, tracker_file_path)