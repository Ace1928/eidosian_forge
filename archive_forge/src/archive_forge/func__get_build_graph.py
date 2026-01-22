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