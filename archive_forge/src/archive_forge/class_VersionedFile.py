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
class VersionedFile:
    """Versioned text file storage.

    A versioned file manages versions of line-based text files,
    keeping track of the originating version for each line.

    To clients the "lines" of the file are represented as a list of
    strings. These strings will typically have terminal newline
    characters, but this is not required.  In particular files commonly
    do not have a newline at the end of the file.

    Texts are identified by a version-id string.
    """

    @staticmethod
    def check_not_reserved_id(version_id):
        revision.check_not_reserved_id(version_id)

    def copy_to(self, name, transport):
        """Copy this versioned file to name on transport."""
        raise NotImplementedError(self.copy_to)

    def get_record_stream(self, versions, ordering, include_delta_closure):
        """Get a stream of records for versions.

        :param versions: The versions to include. Each version is a tuple
            (version,).
        :param ordering: Either 'unordered' or 'topological'. A topologically
            sorted stream has compression parents strictly before their
            children.
        :param include_delta_closure: If True then the closure across any
            compression parents will be included (in the data content of the
            stream, not in the emitted records). This guarantees that
            'fulltext' can be used successfully on every record.
        :return: An iterator of ContentFactory objects, each of which is only
            valid until the iterator is advanced.
        """
        raise NotImplementedError(self.get_record_stream)

    def has_version(self, version_id):
        """Returns whether version is present."""
        raise NotImplementedError(self.has_version)

    def insert_record_stream(self, stream):
        """Insert a record stream into this versioned file.

        :param stream: A stream of records to insert.
        :return: None
        :seealso VersionedFile.get_record_stream:
        """
        raise NotImplementedError

    def add_lines(self, version_id, parents, lines, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False, check_content=True):
        """Add a single text on top of the versioned file.

        Must raise RevisionAlreadyPresent if the new version is
        already present in file history.

        Must raise RevisionNotPresent if any of the given parents are
        not present in file history.

        :param lines: A list of lines. Each line must be a bytestring. And all
            of them except the last must be terminated with 
 and contain no
            other 
's. The last line may either contain no 
's or a single
            terminated 
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
        self._check_write_ok()
        return self._add_lines(version_id, parents, lines, parent_texts, left_matching_blocks, nostore_sha, random_id, check_content)

    def _add_lines(self, version_id, parents, lines, parent_texts, left_matching_blocks, nostore_sha, random_id, check_content):
        """Helper to do the class specific add_lines."""
        raise NotImplementedError(self.add_lines)

    def add_lines_with_ghosts(self, version_id, parents, lines, parent_texts=None, nostore_sha=None, random_id=False, check_content=True, left_matching_blocks=None):
        """Add lines to the versioned file, allowing ghosts to be present.

        This takes the same parameters as add_lines and returns the same.
        """
        self._check_write_ok()
        return self._add_lines_with_ghosts(version_id, parents, lines, parent_texts, nostore_sha, random_id, check_content, left_matching_blocks)

    def _add_lines_with_ghosts(self, version_id, parents, lines, parent_texts, nostore_sha, random_id, check_content, left_matching_blocks):
        """Helper to do class specific add_lines_with_ghosts."""
        raise NotImplementedError(self.add_lines_with_ghosts)

    def check(self, progress_bar=None):
        """Check the versioned file for integrity."""
        raise NotImplementedError(self.check)

    def _check_lines_not_unicode(self, lines):
        """Check that lines being added to a versioned file are not unicode."""
        for line in lines:
            if not isinstance(line, bytes):
                raise errors.BzrBadParameterUnicode('lines')

    def _check_lines_are_lines(self, lines):
        """Check that the lines really are full lines without inline EOL."""
        for line in lines:
            if b'\n' in line[:-1]:
                raise errors.BzrBadParameterContainsNewline('lines')

    def get_format_signature(self):
        """Get a text description of the data encoding in this file.

        :since: 0.90
        """
        raise NotImplementedError(self.get_format_signature)

    def make_mpdiffs(self, version_ids):
        """Create multiparent diffs for specified versions."""
        knit_versions = set()
        knit_versions.update(version_ids)
        parent_map = self.get_parent_map(version_ids)
        for version_id in version_ids:
            try:
                knit_versions.update(parent_map[version_id])
            except KeyError:
                raise errors.RevisionNotPresent(version_id, self)
        knit_versions = set(self.get_parent_map(knit_versions))
        lines = dict(zip(knit_versions, self._get_lf_split_line_list(knit_versions)))
        diffs = []
        for version_id in version_ids:
            target = lines[version_id]
            try:
                parents = [lines[p] for p in parent_map[version_id] if p in knit_versions]
            except KeyError:
                raise errors.RevisionNotPresent(version_id, self)
            if len(parents) > 0:
                left_parent_blocks = self._extract_blocks(version_id, parents[0], target)
            else:
                left_parent_blocks = None
            diffs.append(multiparent.MultiParent.from_lines(target, parents, left_parent_blocks))
        return diffs

    def _extract_blocks(self, version_id, source, target):
        return None

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
        present_parents = set(self.get_parent_map(needed_parents))
        for parent_id, lines in zip(present_parents, self._get_lf_split_line_list(present_parents)):
            mpvf.add_version(lines, parent_id, [])
        for (version, parent_ids, expected_sha1, mpdiff), lines in zip(records, mpvf.get_line_list(versions)):
            if len(parent_ids) == 1:
                left_matching_blocks = list(mpdiff.get_matching_blocks(0, mpvf.get_diff(parent_ids[0]).num_lines()))
            else:
                left_matching_blocks = None
            try:
                _, _, version_text = self.add_lines_with_ghosts(version, parent_ids, lines, vf_parents, left_matching_blocks=left_matching_blocks)
            except NotImplementedError:
                _, _, version_text = self.add_lines(version, parent_ids, lines, vf_parents, left_matching_blocks=left_matching_blocks)
            vf_parents[version] = version_text
        sha1s = self.get_sha1s(versions)
        for version, parent_ids, expected_sha1, mpdiff in records:
            if expected_sha1 != sha1s[version]:
                raise errors.VersionedFileInvalidChecksum(version)

    def get_text(self, version_id):
        """Return version contents as a text string.

        Raises RevisionNotPresent if version is not present in
        file history.
        """
        return b''.join(self.get_lines(version_id))
    get_string = get_text

    def get_texts(self, version_ids):
        """Return the texts of listed versions as a list of strings.

        Raises RevisionNotPresent if version is not present in
        file history.
        """
        return [b''.join(self.get_lines(v)) for v in version_ids]

    def get_lines(self, version_id):
        """Return version contents as a sequence of lines.

        Raises RevisionNotPresent if version is not present in
        file history.
        """
        raise NotImplementedError(self.get_lines)

    def _get_lf_split_line_list(self, version_ids):
        return [BytesIO(t).readlines() for t in self.get_texts(version_ids)]

    def get_ancestry(self, version_ids):
        """Return a list of all ancestors of given version(s). This
        will not include the null revision.

        Must raise RevisionNotPresent if any of the given versions are
        not present in file history."""
        raise NotImplementedError(self.get_ancestry)

    def get_ancestry_with_ghosts(self, version_ids):
        """Return a list of all ancestors of given version(s). This
        will not include the null revision.

        Must raise RevisionNotPresent if any of the given versions are
        not present in file history.

        Ghosts that are known about will be included in ancestry list,
        but are not explicitly marked.
        """
        raise NotImplementedError(self.get_ancestry_with_ghosts)

    def get_parent_map(self, version_ids):
        """Get a map of the parents of version_ids.

        :param version_ids: The version ids to look up parents for.
        :return: A mapping from version id to parents.
        """
        raise NotImplementedError(self.get_parent_map)

    def get_parents_with_ghosts(self, version_id):
        """Return version names for parents of version_id.

        Will raise RevisionNotPresent if version_id is not present
        in the history.

        Ghosts that are known about will be included in the parent list,
        but are not explicitly marked.
        """
        try:
            return list(self.get_parent_map([version_id])[version_id])
        except KeyError:
            raise errors.RevisionNotPresent(version_id, self)

    def annotate(self, version_id):
        """Return a list of (version-id, line) tuples for version_id.

        :raise RevisionNotPresent: If the given version is
        not present in file history.
        """
        raise NotImplementedError(self.annotate)

    def iter_lines_added_or_present_in_versions(self, version_ids=None, pb=None):
        """Iterate over the lines in the versioned file from version_ids.

        This may return lines from other versions. Each item the returned
        iterator yields is a tuple of a line and a text version that that line
        is present in (not introduced in).

        Ordering of results is in whatever order is most suitable for the
        underlying storage format.

        If a progress bar is supplied, it may be used to indicate progress.
        The caller is responsible for cleaning up progress bars (because this
        is an iterator).

        NOTES: Lines are normalised: they will all have 
 terminators.
               Lines are returned in arbitrary order.

        :return: An iterator over (line, version_id).
        """
        raise NotImplementedError(self.iter_lines_added_or_present_in_versions)

    def plan_merge(self, ver_a, ver_b, base=None):
        """Return pseudo-annotation indicating how the two versions merge.

        This is computed between versions a and b and their common
        base.

        Weave lines present in none of them are skipped entirely.

        Legend:
        killed-base Dead in base revision
        killed-both Killed in each revision
        killed-a    Killed in a
        killed-b    Killed in b
        unchanged   Alive in both a and b (possibly created in both)
        new-a       Created in a
        new-b       Created in b
        ghost-a     Killed in a, unborn in b
        ghost-b     Killed in b, unborn in a
        irrelevant  Not in either revision
        """
        raise NotImplementedError(VersionedFile.plan_merge)

    def weave_merge(self, plan, a_marker=TextMerge.A_MARKER, b_marker=TextMerge.B_MARKER):
        return PlanWeaveMerge(plan, a_marker, b_marker).merge_lines()[0]