import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
def _record_to_data(self, key, digest, lines, dense_lines=None):
    """Convert key, digest, lines into a raw data block.

        :param key: The key of the record. Currently keys are always serialised
            using just the trailing component.
        :param dense_lines: The bytes of lines but in a denser form. For
            instance, if lines is a list of 1000 bytestrings each ending in
            \\n, dense_lines may be a list with one line in it, containing all
            the 1000's lines and their \\n's. Using dense_lines if it is
            already known is a win because the string join to create bytes in
            this function spends less time resizing the final string.
        :return: (len, chunked bytestring with compressed data)
        """
    chunks = [b'version %s %d %s\n' % (key[-1], len(lines), digest)]
    chunks.extend(dense_lines or lines)
    chunks.append(b'end ' + key[-1] + b'\n')
    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise AssertionError('data must be plain bytes was %s' % type(chunk))
    if lines and (not lines[-1].endswith(b'\n')):
        raise ValueError('corrupt lines value %r' % lines)
    compressed_chunks = tuned_gzip.chunks_to_gzip(chunks)
    return (sum(map(len, compressed_chunks)), compressed_chunks)