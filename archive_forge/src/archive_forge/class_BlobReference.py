from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
class BlobReference:
    """A reference to a blob.

    Attributes:
      blob_key: A string containing a key uniquely identifying a blob, which
        may be dereferenced via `provider.read_blob(blob_key)`.

        These keys must be constructed such that they can be included directly in
        a URL, with no further encoding. Concretely, this means that they consist
        exclusively of "unreserved characters" per RFC 3986, namely
        [a-zA-Z0-9._~-]. These keys are case-sensitive; it may be wise for
        implementations to normalize case to reduce confusion. The empty string
        is not a valid key.

        Blob keys must not contain information that should be kept secret.
        Privacy-sensitive applications should use random keys (e.g. UUIDs), or
        encrypt keys containing secret fields.
      url: (optional) A string containing a URL from which the blob data may be
        fetched directly, bypassing the data provider. URLs may be a vector
        for data leaks (e.g. via browser history, web proxies, etc.), so these
        URLs should not expose secret information.
    """
    __slots__ = ('_url', '_blob_key')

    def __init__(self, blob_key, url=None):
        self._blob_key = blob_key
        self._url = url

    @property
    def blob_key(self):
        """Provide a key uniquely identifying a blob.

        Callers should consider these keys to be opaque-- i.e., to have
        no intrinsic meaning. Some data providers may use random IDs;
        but others may encode information into the key, in which case
        callers must make no attempt to decode it.
        """
        return self._blob_key

    @property
    def url(self):
        """Provide the direct-access URL for this blob, if available.

        Note that this method is *not* expected to construct a URL to
        the data-loading endpoint provided by TensorBoard. If this
        method returns None, then the caller should proceed to use
        `blob_key()` to build the URL, as needed.
        """
        return self._url

    def __eq__(self, other):
        if not isinstance(other, BlobReference):
            return False
        if self._blob_key != other._blob_key:
            return False
        if self._url != other._url:
            return False
        return True

    def __hash__(self):
        return hash((self._blob_key, self._url))

    def __repr__(self):
        return 'BlobReference(%s)' % ', '.join(('blob_key=%r' % (self._blob_key,), 'url=%r' % (self._url,)))