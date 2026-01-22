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
def NameExpansionIterator(command_name, debug, logger, gsutil_api, url_strs, recursion_requested, all_versions=False, cmd_supports_recursion=True, project_id=None, ignore_symlinks=False, continue_on_error=False, bucket_listing_fields=None):
    """Static factory function for instantiating _NameExpansionIterator.

  This wraps the resulting iterator in a PluralityCheckableIterator and checks
  that it is non-empty. Also, allows url_strs to be either an array or an
  iterator.

  Args:
    command_name: name of command being run.
    debug: Debug level to pass to underlying iterators (range 0..3).
    logger: logging.Logger object.
    gsutil_api: Cloud storage interface.  Settable for testing/mocking.
    url_strs: Iterable URL strings needing expansion.
    recursion_requested: True if -r specified on command-line.  If so,
        listings will be flattened so mapped-to results contain objects
        spanning subdirectories.
    all_versions: Bool indicating whether to iterate over all object versions.
    cmd_supports_recursion: Bool indicating whether this command supports a '-r'
        flag. Useful for printing helpful error messages.
    project_id: Project id to use for the current command.
    ignore_symlinks: If True, ignore symlinks during iteration.
    continue_on_error: If true, yield no-match exceptions encountered during
                       iteration instead of raising them.
    bucket_listing_fields: Iterable fields to include in expanded results.
        Ex. ['name', 'acl']. Underyling iterator is responsible for converting
        these to list-style format ['items/name', 'items/acl']. If this is
        None, only the object name is included in the result.

  Raises:
    CommandException if underlying iterator is empty.

  Returns:
    Name expansion iterator instance.

  For example semantics, see comments in NameExpansionIterator.__init__.
  """
    url_strs = PluralityCheckableIterator(url_strs)
    name_expansion_iterator = _NameExpansionIterator(command_name, debug, logger, gsutil_api, url_strs, recursion_requested, all_versions=all_versions, cmd_supports_recursion=cmd_supports_recursion, project_id=project_id, ignore_symlinks=ignore_symlinks, continue_on_error=continue_on_error, bucket_listing_fields=bucket_listing_fields)
    name_expansion_iterator = PluralityCheckableIterator(name_expansion_iterator)
    if name_expansion_iterator.IsEmpty():
        raise CommandException(NO_URLS_MATCHED_GENERIC)
    return name_expansion_iterator