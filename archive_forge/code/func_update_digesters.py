from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import enum
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import fast_crc32c_util
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
def update_digesters(digesters, data):
    """Updates every hash object with new data in a dict of digesters."""
    for hash_object in digesters.values():
        hash_object.update(data)