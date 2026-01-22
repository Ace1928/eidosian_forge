from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
class BucketListingRef(object):
    """Base class for a reference to one fully expanded iterator result.

  This allows polymorphic iteration over wildcard-iterated URLs.  The
  reference contains a fully expanded URL string containing no wildcards and
  referring to exactly one entity (if a wildcard is contained, it is assumed
  this is part of the raw string and should never be treated as a wildcard).

  Each reference represents a Bucket, Object, or Prefix.  For filesystem URLs,
  Objects represent files and Prefixes represent directories.

  The root_object member contains the underlying object as it was retrieved.
  It is populated by the calling iterator, which may only request certain
  fields to reduce the number of server requests.

  For filesystem URLs, root_object is not populated.
  """

    class _BucketListingRefType(object):
        """Enum class for describing BucketListingRefs."""
        BUCKET = 'bucket'
        OBJECT = 'object'
        PREFIX = 'prefix'

    @property
    def url_string(self):
        return self._url_string

    @property
    def type_name(self):
        return self._ref_type

    def IsBucket(self):
        return self._ref_type == self._BucketListingRefType.BUCKET

    def IsObject(self):
        return self._ref_type == self._BucketListingRefType.OBJECT

    def IsPrefix(self):
        return self._ref_type == self._BucketListingRefType.PREFIX

    def __str__(self):
        return self._url_string