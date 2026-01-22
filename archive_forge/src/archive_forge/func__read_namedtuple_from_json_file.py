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
def _read_namedtuple_from_json_file(named_tuple_class, tracker_file_path):
    """Returns an instance of named_tuple_class with data at tracker_file_path."""
    if not os.path.exists(tracker_file_path):
        return None
    with files.FileReader(tracker_file_path) as tracker_file:
        tracker_dict = json.load(tracker_file)
        return named_tuple_class(**tracker_dict)