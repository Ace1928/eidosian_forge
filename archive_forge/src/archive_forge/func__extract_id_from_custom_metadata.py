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
def _extract_id_from_custom_metadata(resource, key):
    """Finds, validates, and returns a POSIX ID value."""
    if not resource.custom_fields or resource.custom_fields.get(key) is None:
        return None
    try:
        posix_id = int(resource.custom_fields[key])
    except ValueError:
        log.warning('{} metadata did not contain a numeric value for {}: {}'.format(resource.storage_url.url_string, key, resource.custom_fields[key]))
        return None
    if posix_id < 0:
        log.warning('Found negative ID value in {} metadata {}: {}'.format(resource.storage_url.url_string, key, resource.custom_fields[key]))
        return None
    return posix_id