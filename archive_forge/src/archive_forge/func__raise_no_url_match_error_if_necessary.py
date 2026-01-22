from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
def _raise_no_url_match_error_if_necessary(self):
    if not self._raise_error_for_unmatched_urls:
        return
    non_matching_urls = [url for url, found_match in self._url_found_match_tracker.items() if not found_match]
    if non_matching_urls:
        raise errors.InvalidUrlError('The following URLs matched no objects or files:\n-{}'.format('\n-'.join(non_matching_urls)))