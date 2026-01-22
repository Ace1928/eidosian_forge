from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import stat
import sys
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.utils import system_util
from gslib.utils import text_util
class _CloudUrl(StorageUrl):
    """Cloud URL class providing parsing and convenience methods.

    This class assists with usage and manipulation of an
    (optionally wildcarded) cloud URL string.  Depending on the string
    contents, this class represents a provider, bucket(s), or object(s).

    This class operates only on strings.  No cloud storage API calls are
    made from this class.
  """

    def __init__(self, url_string):
        self.scheme = None
        self.delim = '/'
        self.bucket_name = None
        self.object_name = None
        self.generation = None
        provider_match = PROVIDER_REGEX.match(url_string)
        bucket_match = BUCKET_REGEX.match(url_string)
        if provider_match:
            self.scheme = provider_match.group('provider')
        elif bucket_match:
            self.scheme = bucket_match.group('provider')
            self.bucket_name = bucket_match.group('bucket')
        else:
            object_match = OBJECT_REGEX.match(url_string)
            if object_match:
                self.scheme = object_match.group('provider')
                self.bucket_name = object_match.group('bucket')
                self.object_name = object_match.group('object')
                if self.object_name == '.' or self.object_name == '..':
                    raise InvalidUrlError('%s is an invalid root-level object name' % self.object_name)
                if self.scheme == 'gs':
                    generation_match = GS_GENERATION_REGEX.match(self.object_name)
                    if generation_match:
                        self.object_name = generation_match.group('object')
                        self.generation = generation_match.group('generation')
                elif self.scheme == 's3':
                    version_match = S3_VERSION_REGEX.match(self.object_name)
                    if version_match:
                        self.object_name = version_match.group('object')
                        self.generation = version_match.group('version_id')
            else:
                raise InvalidUrlError('CloudUrl: URL string %s did not match URL regex' % url_string)
        if url_string[len(self.scheme) + len('://'):].startswith(self.delim):
            raise InvalidUrlError('Cloud URL scheme should be followed by colon and two slashes: "://". Found: "{}"'.format(url_string))
        self._WarnIfUnsupportedDoubleWildcard()

    def Clone(self):
        return _CloudUrl(self.url_string)

    def IsFileUrl(self):
        return False

    def IsCloudUrl(self):
        return True

    def IsStream(self):
        raise NotImplementedError('IsStream not supported on CloudUrl')

    def IsFifo(self):
        raise NotImplementedError('IsFifo not supported on CloudUrl')

    def IsBucket(self):
        return bool(self.bucket_name and (not self.object_name))

    def IsObject(self):
        return bool(self.bucket_name and self.object_name)

    def HasGeneration(self):
        return bool(self.generation)

    def IsProvider(self):
        return bool(self.scheme and (not self.bucket_name))

    def CreatePrefixUrl(self, wildcard_suffix=None):
        prefix = StripOneSlash(self.versionless_url_string)
        if wildcard_suffix:
            prefix = '%s/%s' % (prefix, wildcard_suffix)
        return prefix

    @property
    def bucket_url_string(self):
        return '%s://%s/' % (self.scheme, self.bucket_name)

    @property
    def url_string(self):
        url_str = self.versionless_url_string
        if self.HasGeneration():
            url_str += '#%s' % self.generation
        return url_str

    @property
    def versionless_url_string(self):
        if self.IsProvider():
            return '%s://' % self.scheme
        elif self.IsBucket():
            return self.bucket_url_string
        return '%s://%s/%s' % (self.scheme, self.bucket_name, self.object_name)

    def __str__(self):
        return self.url_string