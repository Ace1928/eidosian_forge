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
class NameExpansionResult(object):
    """Holds one fully expanded result from iterating over NameExpansionIterator.

  The member data in this class need to be pickleable because
  NameExpansionResult instances are passed through Multiprocessing.Queue. In
  particular, don't include any boto state like StorageUri, since that pulls
  in a big tree of objects, some of which aren't pickleable (and even if
  they were, pickling/unpickling such a large object tree would result in
  significant overhead).

  The state held in this object is needed for handling the various naming cases
  (e.g., copying from a single source URL to a directory generates different
  dest URL names than copying multiple URLs to a directory, to be consistent
  with naming rules used by the Unix cp command). For more details see comments
  in _NameExpansionIterator.
  """

    def __init__(self, source_storage_url, is_multi_source_request, is_multi_top_level_source_request, names_container, expanded_storage_url, expanded_result):
        """Instantiates a result from name expansion.

    Args:
      source_storage_url: StorageUrl that was being expanded.
      is_multi_source_request: bool indicator whether multiple input URLs or
          src_url_str expanded to more than one BucketListingRef.
      is_multi_top_level_source_request: same as is_multi_source_request but
          measured before recursion.
      names_container: Bool indicator whether src_url names a container.
      expanded_storage_url: StorageUrl that was expanded.
      expanded_result: cloud object metadata in MessageToJson form (for
          pickleability), if any was iterated; None otherwise.
          Consumers must call JsonToMessage to get an apitools Object.
    """
        self.source_storage_url = source_storage_url
        self.is_multi_source_request = is_multi_source_request
        self.is_multi_top_level_source_request = is_multi_top_level_source_request
        self.names_container = names_container
        self.expanded_storage_url = expanded_storage_url
        self.expanded_result = encoding.MessageToJson(expanded_result) if expanded_result else None

    def __repr__(self):
        return '%s' % self.expanded_storage_url