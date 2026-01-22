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
class _ImplicitBucketSubdirIterator(object):
    """Iterator wrapper that performs implicit bucket subdir expansion.

  Each iteration yields tuple (names_container, expanded BucketListingRefs)
    where names_container is true if URL names a directory, bucket,
    or bucket subdir.

  For example, iterating over [BucketListingRef("gs://abc")] would expand to:
    [BucketListingRef("gs://abc/o1"), BucketListingRef("gs://abc/o2")]
  if those subdir objects exist, and [BucketListingRef("gs://abc") otherwise.
  """

    def __init__(self, name_exp_instance, blr_iter, subdir_exp_wildcard, bucket_listing_fields):
        """Instantiates the iterator.

    Args:
      name_exp_instance: calling instance of NameExpansion class.
      blr_iter: iterator over BucketListingRef prefixes and objects.
      subdir_exp_wildcard: wildcard for expanding subdirectories;
          expected values are ** if the mapped-to results should contain
          objects spanning subdirectories, or * if only one level should
          be listed.
      bucket_listing_fields: Fields requested in enumerated results.
    """
        self.blr_iter = blr_iter
        self.name_exp_instance = name_exp_instance
        self.subdir_exp_wildcard = subdir_exp_wildcard
        self.bucket_listing_fields = bucket_listing_fields

    def __iter__(self):
        for blr in self.blr_iter:
            if blr.IsPrefix():
                prefix_url = StorageUrlFromString(blr.url_string).CreatePrefixUrl(wildcard_suffix=self.subdir_exp_wildcard)
                implicit_subdir_iterator = PluralityCheckableIterator(self.name_exp_instance.WildcardIterator(prefix_url).IterAll(bucket_listing_fields=self.bucket_listing_fields))
                if not implicit_subdir_iterator.IsEmpty():
                    for exp_blr in implicit_subdir_iterator:
                        yield (True, exp_blr)
                else:
                    yield (False, blr)
            elif blr.IsObject():
                yield (False, blr)
            else:
                raise CommandException('_ImplicitBucketSubdirIterator got a bucket reference %s' % blr)