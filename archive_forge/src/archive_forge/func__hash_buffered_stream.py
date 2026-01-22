import errno
import hashlib
import os.path  # pylint: disable-msg=W0404
import warnings
from typing import Dict, List, Type, Iterator, Optional
from os.path import join as pjoin
import libcloud.utils.files
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import BaseDriver, Connection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.storage.types import ObjectDoesNotExistError
def _hash_buffered_stream(self, stream, hasher, blocksize=65536):
    total_len = 0
    if hasattr(stream, '__next__') or hasattr(stream, 'next'):
        if hasattr(stream, 'seek'):
            try:
                stream.seek(0)
            except OSError as e:
                if e.errno != errno.ESPIPE:
                    raise e
        for chunk in libcloud.utils.files.read_in_chunks(iterator=stream):
            hasher.update(b(chunk))
            total_len += len(chunk)
        return (hasher.hexdigest(), total_len)
    if not hasattr(stream, '__exit__'):
        for s in stream:
            hasher.update(s)
            total_len = total_len + len(s)
        return (hasher.hexdigest(), total_len)
    with stream:
        buf = stream.read(blocksize)
        while len(buf) > 0:
            total_len = total_len + len(buf)
            hasher.update(buf)
            buf = stream.read(blocksize)
    return (hasher.hexdigest(), total_len)