import bz2
import re
from io import BytesIO
import fastbencode as bencode
from .... import errors, iterablefile, lru_cache, multiparent, osutils
from .... import repository as _mod_repository
from .... import revision as _mod_revision
from .... import trace, ui
from ....i18n import ngettext
from ... import pack, serializer
from ... import versionedfile as _mod_versionedfile
from .. import bundle_data
from .. import serializer as bundle_serializer
class BundleWriter:
    """Writer for bundle-format files.

    This serves roughly the same purpose as ContainerReader, but acts as a
    layer on top of it.

    Provides ways of writing the specific record types supported this bundle
    format.
    """

    def __init__(self, fileobj):
        self._container = pack.ContainerWriter(self._write_encoded)
        self._fileobj = fileobj
        self._compressor = bz2.BZ2Compressor()

    def _write_encoded(self, bytes):
        """Write bzip2-encoded bytes to the file"""
        self._fileobj.write(self._compressor.compress(bytes))

    def begin(self):
        """Start writing the bundle"""
        self._fileobj.write(bundle_serializer._get_bundle_header('4'))
        self._fileobj.write(b'#\n')
        self._container.begin()

    def end(self):
        """Finish writing the bundle"""
        self._container.end()
        self._fileobj.write(self._compressor.flush())

    def add_multiparent_record(self, mp_bytes, sha1, parents, repo_kind, revision_id, file_id):
        """Add a record for a multi-parent diff

        :mp_bytes: A multi-parent diff, as a bytestring
        :sha1: The sha1 hash of the fulltext
        :parents: a list of revision-ids of the parents
        :repo_kind: The kind of object in the repository.  May be 'file' or
            'inventory'
        :revision_id: The revision id of the mpdiff being added.
        :file_id: The file-id of the file, or None for inventories.
        """
        metadata = {b'parents': parents, b'storage_kind': b'mpdiff', b'sha1': sha1}
        self._add_record(mp_bytes, metadata, repo_kind, revision_id, file_id)

    def add_fulltext_record(self, bytes, parents, repo_kind, revision_id):
        """Add a record for a fulltext

        :bytes: The fulltext, as a bytestring
        :parents: a list of revision-ids of the parents
        :repo_kind: The kind of object in the repository.  May be 'revision' or
            'signature'
        :revision_id: The revision id of the fulltext being added.
        """
        metadata = {b'parents': parents, b'storage_kind': b'mpdiff'}
        self._add_record(bytes, {b'parents': parents, b'storage_kind': b'fulltext'}, repo_kind, revision_id, None)

    def add_info_record(self, kwargs):
        """Add an info record to the bundle

        Any parameters may be supplied, except 'self' and 'storage_kind'.
        Values must be lists, strings, integers, dicts, or a combination.
        """
        kwargs[b'storage_kind'] = b'header'
        self._add_record(None, kwargs, 'info', None, None)

    @staticmethod
    def encode_name(content_kind, revision_id, file_id=None):
        """Encode semantic ids as a container name"""
        if content_kind not in ('revision', 'file', 'inventory', 'signature', 'info'):
            raise ValueError(content_kind)
        if content_kind == 'file':
            if file_id is None:
                raise AssertionError()
        elif file_id is not None:
            raise AssertionError()
        if content_kind == 'info':
            if revision_id is not None:
                raise AssertionError()
        elif revision_id is None:
            raise AssertionError()
        names = [n.replace(b'/', b'//') for n in (content_kind.encode('ascii'), revision_id, file_id) if n is not None]
        return b'/'.join(names)

    def _add_record(self, bytes, metadata, repo_kind, revision_id, file_id):
        """Add a bundle record to the container.

        Most bundle records are recorded as header/body pairs, with the
        body being nameless.  Records with storage_kind 'header' have no
        body.
        """
        name = self.encode_name(repo_kind, revision_id, file_id)
        encoded_metadata = bencode.bencode(metadata)
        self._container.add_bytes_record([encoded_metadata], len(encoded_metadata), [(name,)])
        if metadata[b'storage_kind'] != b'header':
            self._container.add_bytes_record([bytes], len(bytes), [])