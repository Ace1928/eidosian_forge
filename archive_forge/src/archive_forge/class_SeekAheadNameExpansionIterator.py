from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import logging
import os
import sys
import six
from apitools.base.py import encoding
import gslib
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_GENERIC
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
from gslib.seek_ahead_thread import SeekAheadResult
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
import gslib.wildcard_iterator
from gslib.wildcard_iterator import StorageUrlFromString
class SeekAheadNameExpansionIterator(object):
    """Creates and wraps a _NameExpansionIterator and yields SeekAheadResults.

  Unlike the NameExpansionIterator, which can make API calls upon __init__
  to check for plurality, this iterator does no work until the first iteration
  occurs.
  """

    def __init__(self, command_name, debug, gsutil_api, url_strs, recursion_requested, all_versions=False, cmd_supports_recursion=True, project_id=None, ignore_symlinks=False, file_size_will_change=False):
        """Initializes a _NameExpansionIterator with the inputs."""
        self.count_data_bytes = command_name in ('cp', 'mv', 'rewrite') and (not file_size_will_change)
        bucket_listing_fields = ['size'] if self.count_data_bytes else None
        self.name_expansion_iterator = _NameExpansionIterator(command_name, debug, logging.getLogger('dummy'), gsutil_api, PluralityCheckableIterator(url_strs), recursion_requested, all_versions=all_versions, cmd_supports_recursion=cmd_supports_recursion, project_id=project_id, ignore_symlinks=ignore_symlinks, continue_on_error=True, bucket_listing_fields=bucket_listing_fields)

    def __iter__(self):
        for name_expansion_result in self.name_expansion_iterator:
            if self.count_data_bytes and name_expansion_result.expanded_result:
                iterated_metadata = encoding.JsonToMessage(apitools_messages.Object, name_expansion_result.expanded_result)
                iterated_size = iterated_metadata.size or 0
                yield SeekAheadResult(data_bytes=iterated_size)
            else:
                yield SeekAheadResult()