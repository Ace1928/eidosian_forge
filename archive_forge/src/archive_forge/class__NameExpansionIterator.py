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
class _NameExpansionIterator(object):
    """Class that iterates over all source URLs passed to the iterator.

  See details in __iter__ function doc.
  """

    def __init__(self, command_name, debug, logger, gsutil_api, url_strs, recursion_requested, all_versions=False, cmd_supports_recursion=True, project_id=None, ignore_symlinks=False, continue_on_error=False, bucket_listing_fields=None):
        """Creates a NameExpansionIterator.

    Args:
      command_name: name of command being run.
      debug: Debug level to pass to underlying iterators (range 0..3).
      logger: logging.Logger object.
      gsutil_api: Cloud storage interface.  Settable for testing/mocking.
      url_strs: PluralityCheckableIterator of URL strings needing expansion.
      recursion_requested: True if -r specified on command-line.  If so,
          listings will be flattened so mapped-to results contain objects
          spanning subdirectories.
      all_versions: Bool indicating whether to iterate over all object versions.
      cmd_supports_recursion: Bool indicating whether this command supports a
          '-r' flag. Useful for printing helpful error messages.
      project_id: Project id to use for bucket retrieval.
      ignore_symlinks: If True, ignore symlinks during iteration.
      continue_on_error: If true, yield no-match exceptions encountered during
                         iteration instead of raising them.
      bucket_listing_fields: Iterable fields to include in expanded results.
          Ex. ['name', 'acl']. Underyling iterator is responsible for converting
          these to list-style format ['items/name', 'items/acl']. If this is
          None, only the object name is included in the result.

    Examples of _NameExpansionIterator with recursion_requested=True:
      - Calling with one of the url_strs being 'gs://bucket' will enumerate all
        top-level objects, as will 'gs://bucket/' and 'gs://bucket/*'.
      - 'gs://bucket/**' will enumerate all objects in the bucket.
      - 'gs://bucket/abc' will enumerate either the single object abc or, if
         abc is a subdirectory, all objects under abc and any of its
         subdirectories.
      - 'gs://bucket/abc/**' will enumerate all objects under abc or any of its
        subdirectories.
      - 'file:///tmp' will enumerate all files under /tmp, as will
        'file:///tmp/*'
      - 'file:///tmp/**' will enumerate all files under /tmp or any of its
        subdirectories.

    Example if recursion_requested=False:
      calling with gs://bucket/abc/* lists matching objects
      or subdirs, but not sub-subdirs or objects beneath subdirs.

    Note: In step-by-step comments below we give examples assuming there's a
    gs://bucket with object paths:
      abcd/o1.txt
      abcd/o2.txt
      xyz/o1.txt
      xyz/o2.txt
    and a directory file://dir with file paths:
      dir/a.txt
      dir/b.txt
      dir/c/
    """
        self.command_name = command_name
        self.debug = debug
        self.logger = logger
        self.gsutil_api = gsutil_api
        self.url_strs = url_strs
        self.recursion_requested = recursion_requested
        self.all_versions = all_versions
        self.url_strs.has_plurality = self.url_strs.HasPlurality()
        self.cmd_supports_recursion = cmd_supports_recursion
        self.project_id = project_id
        self.ignore_symlinks = ignore_symlinks
        self.continue_on_error = continue_on_error
        self.bucket_listing_fields = set(['name']) if not bucket_listing_fields else bucket_listing_fields
        self._flatness_wildcard = {True: '**', False: '*'}

    def __iter__(self):
        """Iterates over all source URLs passed to the iterator.

    For each src url, expands wildcards, object-less bucket names,
    subdir bucket names, and directory names, and generates a flat listing of
    all the matching objects/files.

    You should instantiate this object using the static factory function
    NameExpansionIterator, because consumers of this iterator need the
    PluralityCheckableIterator wrapper built by that function.

    Yields:
      gslib.name_expansion.NameExpansionResult.

    Raises:
      CommandException: if errors encountered.
    """
        for url_str in self.url_strs:
            storage_url = StorageUrlFromString(url_str)
            if storage_url.IsFileUrl() and (storage_url.IsStream() or storage_url.IsFifo()):
                if self.url_strs.has_plurality:
                    raise CommandException('Multiple URL strings are not supported with streaming ("-") URLs or named pipes.')
                yield NameExpansionResult(source_storage_url=storage_url, is_multi_source_request=False, is_multi_top_level_source_request=False, names_container=False, expanded_storage_url=storage_url, expanded_result=None)
                continue
            src_names_bucket = False
            if storage_url.IsCloudUrl() and storage_url.IsBucket() and (not self.recursion_requested):
                post_step1_iter = PluralityCheckableIterator(self.WildcardIterator(url_str).IterBuckets(bucket_fields=['id']))
            else:
                post_step1_iter = PluralityCheckableIterator(self.WildcardIterator(url_str).IterAll(bucket_listing_fields=self.bucket_listing_fields, expand_top_level_buckets=True))
                if storage_url.IsCloudUrl() and storage_url.IsBucket():
                    src_names_bucket = True
            src_url_expands_to_multi = post_step1_iter.HasPlurality()
            is_multi_top_level_source_request = self.url_strs.has_plurality or src_url_expands_to_multi
            subdir_exp_wildcard = self._flatness_wildcard[self.recursion_requested]
            if self.recursion_requested:
                post_step2_iter = _ImplicitBucketSubdirIterator(self, post_step1_iter, subdir_exp_wildcard, self.bucket_listing_fields)
            else:
                post_step2_iter = _NonContainerTuplifyIterator(post_step1_iter)
            post_step2_iter = PluralityCheckableIterator(post_step2_iter)
            if post_step2_iter.IsEmpty():
                if self.continue_on_error:
                    try:
                        raise CommandException(NO_URLS_MATCHED_TARGET % url_str)
                    except CommandException as e:
                        yield (e, sys.exc_info()[2])
                else:
                    raise CommandException(NO_URLS_MATCHED_TARGET % url_str)
            post_step3_iter = PluralityCheckableIterator(_OmitNonRecursiveIterator(post_step2_iter, self.recursion_requested, self.command_name, self.cmd_supports_recursion, self.logger))
            src_url_expands_to_multi = post_step3_iter.HasPlurality()
            is_multi_source_request = self.url_strs.has_plurality or src_url_expands_to_multi
            for names_container, blr in post_step3_iter:
                src_names_container = src_names_bucket or names_container
                if blr.IsObject():
                    yield NameExpansionResult(source_storage_url=storage_url, is_multi_source_request=is_multi_source_request, is_multi_top_level_source_request=is_multi_top_level_source_request, names_container=src_names_container, expanded_storage_url=blr.storage_url, expanded_result=blr.root_object)
                else:
                    expanded_url = StorageUrlFromString(blr.url_string)
                    if expanded_url.IsFileUrl():
                        url_to_iterate = '%s%s%s' % (blr, os.sep, subdir_exp_wildcard)
                    else:
                        url_to_iterate = expanded_url.CreatePrefixUrl(wildcard_suffix=subdir_exp_wildcard)
                    wc_iter = PluralityCheckableIterator(self.WildcardIterator(url_to_iterate).IterObjects(bucket_listing_fields=self.bucket_listing_fields))
                    src_url_expands_to_multi = src_url_expands_to_multi or wc_iter.HasPlurality()
                    is_multi_source_request = self.url_strs.has_plurality or src_url_expands_to_multi
                    for blr in wc_iter:
                        yield NameExpansionResult(source_storage_url=storage_url, is_multi_source_request=is_multi_source_request, is_multi_top_level_source_request=is_multi_top_level_source_request, names_container=True, expanded_storage_url=blr.storage_url, expanded_result=blr.root_object)

    def WildcardIterator(self, url_string):
        """Helper to instantiate gslib.WildcardIterator.

    Args are same as gslib.WildcardIterator interface, but this method fills
    in most of the values from instance state.

    Args:
      url_string: URL string naming wildcard objects to iterate.

    Returns:
      Wildcard iterator over URL string.
    """
        return gslib.wildcard_iterator.CreateWildcardIterator(url_string, self.gsutil_api, all_versions=self.all_versions, project_id=self.project_id, ignore_symlinks=self.ignore_symlinks, logger=self.logger)