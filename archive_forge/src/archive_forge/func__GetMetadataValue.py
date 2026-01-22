from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.os_config.troubleshoot import utils
def _GetMetadataValue(metadata, key):
    """Gets the value of the key field of the given metadata list.

  Args:
    metadata: The metadata to look through.
    key: the key to look for

  Returns:
  The value of the key, None if the metadata field does not exist.
  """
    return next((md.value for md in metadata if md.key == key), None)