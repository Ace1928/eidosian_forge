import atexit
import errno
import os
import re
import shutil
import sys
import tempfile
from hashlib import md5
from io import BytesIO
from json import dumps
from time import sleep
from httplib2 import Http, urlnorm
from wadllib.application import Application
from lazr.restfulclient._json import DatetimeJSONEncoder
from lazr.restfulclient.errors import HTTPError, error_for
from lazr.uri import URI
class MultipleRepresentationCache(AtomicFileCache):
    """A cache that can hold different representations of the same resource.

    If a resource has two representations with two media types,
    FileCache will only store the most recently fetched
    representation. This cache can keep track of multiple
    representations of the same resource.

    This class works on the assumption that outside calling code sets
    an instance's request_media_type attribute to the value of the
    'Accept' header before initiating the request.

    This class is very much not thread-safe, but FileCache isn't
    thread-safe anyway.
    """

    def __init__(self, cache):
        """Tell FileCache to call append_media_type when generating keys."""
        super(MultipleRepresentationCache, self).__init__(cache, self.append_media_type)
        self.request_media_type = None

    def append_media_type(self, key):
        """Append the request media type to the cache key.

        This ensures that representations of the same resource will be
        cached separately, so long as they're served as different
        media types.
        """
        if self.request_media_type is not None:
            key = key + '-' + self.request_media_type
        return safename(key)

    def _getCachedHeader(self, uri, header):
        """Retrieve a cached value for an HTTP header."""
        scheme, authority, request_uri, cachekey = urlnorm(uri)
        cached_value = self.get(cachekey)
        header_start = header + ':'
        if not isinstance(header_start, bytes):
            header_start = header_start.encode('utf-8')
        if cached_value is not None:
            for line in BytesIO(cached_value):
                if line.startswith(header_start):
                    return line[len(header_start):].strip()
        return None