import itertools
import os
import struct
from copy import copy
from io import BytesIO
from typing import Any, Tuple
from zlib import adler32
from ..lazy_import import lazy_import
import fastbencode as bencode
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import graph as _mod_graph
from .. import osutils
from .. import transport as _mod_transport
from ..registry import Registry
from ..textmerge import TextMerge
from . import index
class VersionedFiles:
    """Storage for many versioned files.

    This object allows a single keyspace for accessing the history graph and
    contents of named bytestrings.

    Currently no implementation allows the graph of different key prefixes to
    intersect, but the API does allow such implementations in the future.

    The keyspace is expressed via simple tuples. Any instance of VersionedFiles
    may have a different length key-size, but that size will be constant for
    all texts added to or retrieved from it. For instance, breezy uses
    instances with a key-size of 2 for storing user files in a repository, with
    the first element the fileid, and the second the version of that file.

    The use of tuples allows a single code base to support several different
    uses with only the mapping logic changing from instance to instance.

    :ivar _immediate_fallback_vfs: For subclasses that support stacking,
        this is a list of other VersionedFiles immediately underneath this
        one.  They may in turn each have further fallbacks.
    """

    def add_lines(self, key, parents, lines, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False, check_content=True):
        """Add a text to the store.

        :param key: The key tuple of the text to add. If the last element is
            None, a CHK string will be generated during the addition.
        :param parents: The parents key tuples of the text to add.
        :param lines: A list of lines. Each line must be a bytestring. And all
            of them except the last must be terminated with 
 and contain no
            other 
's. The last line may either contain no 
's or a single
            terminating 
. If the lines list does meet this constraint the add
            routine may error or may succeed - but you will be unable to read
            the data back accurately. (Checking the lines have been split
            correctly is expensive and extremely unlikely to catch bugs so it
            is not done at runtime unless check_content is True.)
        :param parent_texts: An optional dictionary containing the opaque
            representations of some or all of the parents of version_id to
            allow delta optimisations.  VERY IMPORTANT: the texts must be those
            returned by add_lines or data corruption can be caused.
        :param left_matching_blocks: a hint about which areas are common
            between the text and its left-hand-parent.  The format is
            the SequenceMatcher.get_matching_blocks format.
        :param nostore_sha: Raise ExistingContent and do not add the lines to
            the versioned file if the digest of the lines matches this.
        :param random_id: If True a random id has been selected rather than
            an id determined by some deterministic process such as a converter
            from a foreign VCS. When True the backend may choose not to check
            for uniqueness of the resulting key within the versioned file, so
            this should only be done when the result is expected to be unique
            anyway.
        :param check_content: If True, the lines supplied are verified to be
            bytestrings that are correctly formed lines.
        :return: The text sha1, the number of bytes in the text, and an opaque
                 representation of the inserted version which can be provided
                 back to future add_lines calls in the parent_texts dictionary.
        """
        raise NotImplementedError(self.add_lines)

    def add_content(self, factory, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False, check_content=True):
        """Add a text to the store from a chunk iterable.

        :param key: The key tuple of the text to add. If the last element is
            None, a CHK string will be generated during the addition.
        :param parents: The parents key tuples of the text to add.
        :param chunk_iter: An iterable over bytestrings.
        :param parent_texts: An optional dictionary containing the opaque
            representations of some or all of the parents of version_id to
            allow delta optimisations.  VERY IMPORTANT: the texts must be those
            returned by add_lines or data corruption can be caused.
        :param left_matching_blocks: a hint about which areas are common
            between the text and its left-hand-parent.  The format is
            the SequenceMatcher.get_matching_blocks format.
        :param nostore_sha: Raise ExistingContent and do not add the lines to
            the versioned file if the digest of the lines matches this.
        :param random_id: If True a random id has been selected rather than
            an id determined by some deterministic process such as a converter
            from a foreign VCS. When True the backend may choose not to check
            for uniqueness of the resulting key within the versioned file, so
            this should only be done when the result is expected to be unique
            anyway.
        :param check_content: If True, the lines supplied are verified to be
            bytestrings that are correctly formed lines.
        :return: The text sha1, the number of bytes in the text, and an opaque
                 representation of the inserted version which can be provided
                 back to future add_lines calls in the parent_texts dictionary.
        """
        raise NotImplementedError(self.add_content)

    def add_mpdiffs(self, records):
        """Add mpdiffs to this VersionedFile.

        Records should be iterables of version, parents, expected_sha1,
        mpdiff. mpdiff should be a MultiParent instance.
        """
        vf_parents = {}
        mpvf = multiparent.MultiMemoryVersionedFile()
        versions = []
        for version, parent_ids, expected_sha1, mpdiff in records:
            versions.append(version)
            mpvf.add_diff(mpdiff, version, parent_ids)
        needed_parents = set()
        for version, parent_ids, expected_sha1, mpdiff in records:
            needed_parents.update((p for p in parent_ids if not mpvf.has_version(p)))
        for record in self.get_record_stream(needed_parents, 'unordered', True):
            if record.storage_kind == 'absent':
                continue
            mpvf.add_version(record.get_bytes_as('lines'), record.key, [])
        for (key, parent_keys, expected_sha1, mpdiff), lines in zip(records, mpvf.get_line_list(versions)):
            if len(parent_keys) == 1:
                left_matching_blocks = list(mpdiff.get_matching_blocks(0, mpvf.get_diff(parent_keys[0]).num_lines()))
            else:
                left_matching_blocks = None
            version_sha1, _, version_text = self.add_lines(key, parent_keys, lines, vf_parents, left_matching_blocks=left_matching_blocks)
            if version_sha1 != expected_sha1:
                raise errors.VersionedFileInvalidChecksum(version)
            vf_parents[key] = version_text

    def annotate(self, key):
        """Return a list of (version-key, line) tuples for the text of key.

        :raise RevisionNotPresent: If the key is not present.
        """
        raise NotImplementedError(self.annotate)

    def check(self, progress_bar=None):
        """Check this object for integrity.

        :param progress_bar: A progress bar to output as the check progresses.
        :param keys: Specific keys within the VersionedFiles to check. When
            this parameter is not None, check() becomes a generator as per
            get_record_stream. The difference to get_record_stream is that
            more or deeper checks will be performed.
        :return: None, or if keys was supplied a generator as per
            get_record_stream.
        """
        raise NotImplementedError(self.check)

    @staticmethod
    def check_not_reserved_id(version_id):
        revision.check_not_reserved_id(version_id)

    def clear_cache(self):
        """Clear whatever caches this VersionedFile holds.

        This is generally called after an operation has been performed, when we
        don't expect to be using this versioned file again soon.
        """

    def _check_lines_not_unicode(self, lines):
        """Check that lines being added to a versioned file are not unicode."""
        for line in lines:
            if line.__class__ is not bytes:
                raise errors.BzrBadParameterUnicode('lines')

    def _check_lines_are_lines(self, lines):
        """Check that the lines really are full lines without inline EOL."""
        for line in lines:
            if b'\n' in line[:-1]:
                raise errors.BzrBadParameterContainsNewline('lines')

    def get_known_graph_ancestry(self, keys):
        """Get a KnownGraph instance with the ancestry of keys."""
        pending = set(keys)
        parent_map = {}
        while pending:
            this_parent_map = self.get_parent_map(pending)
            parent_map.update(this_parent_map)
            pending = set(itertools.chain.from_iterable(this_parent_map.values()))
            pending.difference_update(parent_map)
        kg = _mod_graph.KnownGraph(parent_map)
        return kg

    def get_parent_map(self, keys):
        """Get a map of the parents of keys.

        :param keys: The keys to look up parents for.
        :return: A mapping from keys to parents. Absent keys are absent from
            the mapping.
        """
        raise NotImplementedError(self.get_parent_map)

    def get_record_stream(self, keys, ordering, include_delta_closure):
        """Get a stream of records for keys.

        :param keys: The keys to include.
        :param ordering: Either 'unordered' or 'topological'. A topologically
            sorted stream has compression parents strictly before their
            children.
        :param include_delta_closure: If True then the closure across any
            compression parents will be included (in the opaque data).
        :return: An iterator of ContentFactory objects, each of which is only
            valid until the iterator is advanced.
        """
        raise NotImplementedError(self.get_record_stream)

    def get_sha1s(self, keys):
        """Get the sha1's of the texts for the given keys.

        :param keys: The names of the keys to lookup
        :return: a dict from key to sha1 digest. Keys of texts which are not
            present in the store are not present in the returned
            dictionary.
        """
        raise NotImplementedError(self.get_sha1s)
    __contains__ = index._has_key_from_parent_map

    def get_missing_compression_parent_keys(self):
        """Return an iterable of keys of missing compression parents.

        Check this after calling insert_record_stream to find out if there are
        any missing compression parents.  If there are, the records that
        depend on them are not able to be inserted safely. The precise
        behaviour depends on the concrete VersionedFiles class in use.

        Classes that do not support this will raise NotImplementedError.
        """
        raise NotImplementedError(self.get_missing_compression_parent_keys)

    def insert_record_stream(self, stream):
        """Insert a record stream into this container.

        :param stream: A stream of records to insert.
        :return: None
        :seealso VersionedFile.get_record_stream:
        """
        raise NotImplementedError

    def iter_lines_added_or_present_in_keys(self, keys, pb=None):
        """Iterate over the lines in the versioned files from keys.

        This may return lines from other keys. Each item the returned
        iterator yields is a tuple of a line and a text version that that line
        is present in (not introduced in).

        Ordering of results is in whatever order is most suitable for the
        underlying storage format.

        If a progress bar is supplied, it may be used to indicate progress.
        The caller is responsible for cleaning up progress bars (because this
        is an iterator).

        NOTES:
         * Lines are normalised by the underlying store: they will all have 

           terminators.
         * Lines are returned in arbitrary order.

        :return: An iterator over (line, key).
        """
        raise NotImplementedError(self.iter_lines_added_or_present_in_keys)

    def keys(self):
        """Return a iterable of the keys for all the contained texts."""
        raise NotImplementedError(self.keys)

    def make_mpdiffs(self, keys):
        """Create multiparent diffs for specified keys."""
        generator = _MPDiffGenerator(self, keys)
        return generator.compute_diffs()

    def get_annotator(self):
        from ..annotate import Annotator
        return Annotator(self)
    missing_keys = index._missing_keys_from_parent_map

    def _extract_blocks(self, version_id, source, target):
        return None

    def _transitive_fallbacks(self):
        """Return the whole stack of fallback versionedfiles.

        This VersionedFiles may have a list of fallbacks, but it doesn't
        necessarily know about the whole stack going down, and it can't know
        at open time because they may change after the objects are opened.
        """
        all_fallbacks = []
        for a_vfs in self._immediate_fallback_vfs:
            all_fallbacks.append(a_vfs)
            all_fallbacks.extend(a_vfs._transitive_fallbacks())
        return all_fallbacks