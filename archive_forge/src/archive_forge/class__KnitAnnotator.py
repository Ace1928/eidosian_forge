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
class _KnitAnnotator(annotate.Annotator):
    """Build up the annotations for a text."""

    def __init__(self, vf):
        annotate.Annotator.__init__(self, vf)
        self._matching_blocks = {}
        self._content_objects = {}
        self._num_compression_children = {}
        self._pending_deltas = {}
        self._pending_annotation = {}
        self._all_build_details = {}

    def _get_build_graph(self, key):
        """Get the graphs for building texts and annotations.

        The data you need for creating a full text may be different than the
        data you need to annotate that text. (At a minimum, you need both
        parents to create an annotation, but only need 1 parent to generate the
        fulltext.)

        :return: A list of (key, index_memo) records, suitable for
            passing to read_records_iter to start reading in the raw data from
            the pack file.
        """
        pending = {key}
        records = []
        ann_keys = set()
        self._num_needed_children[key] = 1
        while pending:
            this_iteration = pending
            build_details = self._vf._index.get_build_details(this_iteration)
            self._all_build_details.update(build_details)
            pending = set()
            for key, details in build_details.items():
                index_memo, compression_parent, parent_keys, record_details = details
                self._parent_map[key] = parent_keys
                self._heads_provider = None
                records.append((key, index_memo))
                pending.update([p for p in parent_keys if p not in self._all_build_details])
                if parent_keys:
                    for parent_key in parent_keys:
                        if parent_key in self._num_needed_children:
                            self._num_needed_children[parent_key] += 1
                        else:
                            self._num_needed_children[parent_key] = 1
                if compression_parent:
                    if compression_parent in self._num_compression_children:
                        self._num_compression_children[compression_parent] += 1
                    else:
                        self._num_compression_children[compression_parent] = 1
            missing_versions = this_iteration.difference(build_details)
            if missing_versions:
                for key in missing_versions:
                    if key in self._parent_map and key in self._text_cache:
                        ann_keys.add(key)
                        parent_keys = self._parent_map[key]
                        for parent_key in parent_keys:
                            if parent_key in self._num_needed_children:
                                self._num_needed_children[parent_key] += 1
                            else:
                                self._num_needed_children[parent_key] = 1
                        pending.update([p for p in parent_keys if p not in self._all_build_details])
                    else:
                        raise errors.RevisionNotPresent(key, self._vf)
        records.reverse()
        return (records, ann_keys)

    def _get_needed_texts(self, key, pb=None):
        if len(self._vf._immediate_fallback_vfs) > 0:
            yield from annotate.Annotator._get_needed_texts(self, key, pb=pb)
            return
        while True:
            try:
                records, ann_keys = self._get_build_graph(key)
                for idx, (sub_key, text, num_lines) in enumerate(self._extract_texts(records)):
                    if pb is not None:
                        pb.update(gettext('annotating'), idx, len(records))
                    yield (sub_key, text, num_lines)
                for sub_key in ann_keys:
                    text = self._text_cache[sub_key]
                    num_lines = len(text)
                    yield (sub_key, text, num_lines)
                return
            except pack_repo.RetryWithNewPacks as e:
                self._vf._access.reload_or_raise(e)
                self._all_build_details.clear()

    def _cache_delta_blocks(self, key, compression_parent, delta, lines):
        parent_lines = self._text_cache[compression_parent]
        blocks = list(KnitContent.get_line_delta_blocks(delta, parent_lines, lines))
        self._matching_blocks[key, compression_parent] = blocks

    def _expand_record(self, key, parent_keys, compression_parent, record, record_details):
        delta = None
        if compression_parent:
            if compression_parent not in self._content_objects:
                self._pending_deltas.setdefault(compression_parent, []).append((key, parent_keys, record, record_details))
                return None
            num = self._num_compression_children[compression_parent]
            num -= 1
            if num == 0:
                base_content = self._content_objects.pop(compression_parent)
                self._num_compression_children.pop(compression_parent)
            else:
                self._num_compression_children[compression_parent] = num
                base_content = self._content_objects[compression_parent]
            content, delta = self._vf._factory.parse_record(key, record, record_details, base_content, copy_base_content=True)
        else:
            content, _ = self._vf._factory.parse_record(key, record, record_details, None)
        if self._num_compression_children.get(key, 0) > 0:
            self._content_objects[key] = content
        lines = content.text()
        self._text_cache[key] = lines
        if delta is not None:
            self._cache_delta_blocks(key, compression_parent, delta, lines)
        return lines

    def _get_parent_annotations_and_matches(self, key, text, parent_key):
        """Get the list of annotations for the parent, and the matching lines.

        :param text: The opaque value given by _get_needed_texts
        :param parent_key: The key for the parent text
        :return: (parent_annotations, matching_blocks)
            parent_annotations is a list as long as the number of lines in
                parent
            matching_blocks is a list of (parent_idx, text_idx, len) tuples
                indicating which lines match between the two texts
        """
        block_key = (key, parent_key)
        if block_key in self._matching_blocks:
            blocks = self._matching_blocks.pop(block_key)
            parent_annotations = self._annotations_cache[parent_key]
            return (parent_annotations, blocks)
        return annotate.Annotator._get_parent_annotations_and_matches(self, key, text, parent_key)

    def _process_pending(self, key):
        """The content for 'key' was just processed.

        Determine if there is any more pending work to be processed.
        """
        to_return = []
        if key in self._pending_deltas:
            compression_parent = key
            children = self._pending_deltas.pop(key)
            for child_key, parent_keys, record, record_details in children:
                lines = self._expand_record(child_key, parent_keys, compression_parent, record, record_details)
                if self._check_ready_for_annotations(child_key, parent_keys):
                    to_return.append(child_key)
        if key in self._pending_annotation:
            children = self._pending_annotation.pop(key)
            to_return.extend([c for c, p_keys in children if self._check_ready_for_annotations(c, p_keys)])
        return to_return

    def _check_ready_for_annotations(self, key, parent_keys):
        """return true if this text is ready to be yielded.

        Otherwise, this will return False, and queue the text into
        self._pending_annotation
        """
        for parent_key in parent_keys:
            if parent_key not in self._annotations_cache:
                self._pending_annotation.setdefault(parent_key, []).append((key, parent_keys))
                return False
        return True

    def _extract_texts(self, records):
        """Extract the various texts needed based on records"""
        pending_deltas = {}
        for key, record, digest in self._vf._read_records_iter(records):
            details = self._all_build_details[key]
            _, compression_parent, parent_keys, record_details = details
            lines = self._expand_record(key, parent_keys, compression_parent, record, record_details)
            if lines is None:
                continue
            yield_this_text = self._check_ready_for_annotations(key, parent_keys)
            if yield_this_text:
                yield (key, lines, len(lines))
            to_process = self._process_pending(key)
            while to_process:
                this_process = to_process
                to_process = []
                for key in this_process:
                    lines = self._text_cache[key]
                    yield (key, lines, len(lines))
                    to_process.extend(self._process_pending(key))