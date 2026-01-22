from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import os
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.cache import function_result_cache
from googlecloudsdk.core.util import platforms
def _extract_mode_from_custom_metadata(resource):
    """Finds, validates, and returns a POSIX mode value."""
    if not resource.custom_fields or resource.custom_fields.get(MODE_METADATA_KEY) is None:
        return None
    try:
        return PosixMode.from_base_eight_str(resource.custom_fields[MODE_METADATA_KEY])
    except ValueError:
        log.warning('{} metadata did not contain a valid permissions octal string for {}: {}'.format(resource.storage_url.url_string, MODE_METADATA_KEY, resource.custom_fields[MODE_METADATA_KEY]))
    return None