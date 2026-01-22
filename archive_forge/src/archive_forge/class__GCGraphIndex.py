import time
import zlib
from typing import Type
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import errors, osutils, trace
from ..lru_cache import LRUSizeCache
from .btree_index import BTreeBuilder
from .versionedfile import (AbsentContentFactory, ChunkedContentFactory,
from ._groupcompress_py import (LinesDeltaIndex, apply_delta,
class _GCGraphIndex:
    """Mapper from GroupCompressVersionedFiles needs into GraphIndex storage."""

    def __init__(self, graph_index, is_locked, parents=True, add_callback=None, track_external_parent_refs=False, inconsistency_fatal=True, track_new_keys=False):
        """Construct a _GCGraphIndex on a graph_index.

        :param graph_index: An implementation of breezy.index.GraphIndex.
        :param is_locked: A callback, returns True if the index is locked and
            thus usable.
        :param parents: If True, record knits parents, if not do not record
            parents.
        :param add_callback: If not None, allow additions to the index and call
            this callback with a list of added GraphIndex nodes:
            [(node, value, node_refs), ...]
        :param track_external_parent_refs: As keys are added, keep track of the
            keys they reference, so that we can query get_missing_parents(),
            etc.
        :param inconsistency_fatal: When asked to add records that are already
            present, and the details are inconsistent with the existing
            record, raise an exception instead of warning (and skipping the
            record).
        """
        self._add_callback = add_callback
        self._graph_index = graph_index
        self._parents = parents
        self.has_graph = parents
        self._is_locked = is_locked
        self._inconsistency_fatal = inconsistency_fatal
        self._int_cache = {}
        if track_external_parent_refs:
            self._key_dependencies = _KeyRefs(track_new_keys=track_new_keys)
        else:
            self._key_dependencies = None

    def add_records(self, records, random_id=False):
        """Add multiple records to the index.

        This function does not insert data into the Immutable GraphIndex
        backing the KnitGraphIndex, instead it prepares data for insertion by
        the caller and checks that it is safe to insert then calls
        self._add_callback with the prepared GraphIndex nodes.

        :param records: a list of tuples:
                         (key, options, access_memo, parents).
        :param random_id: If True the ids being added were randomly generated
            and no check for existence will be performed.
        """
        if not self._add_callback:
            raise errors.ReadOnlyError(self)
        changed = False
        keys = {}
        for key, value, refs in records:
            if not self._parents:
                if refs:
                    for ref in refs:
                        if ref:
                            raise knit.KnitCorrupt(self, 'attempt to add node with parents in parentless index.')
                    refs = ()
                    changed = True
            keys[key] = (value, refs)
        if not random_id:
            present_nodes = self._get_entries(keys)
            for index, key, value, node_refs in present_nodes:
                node_refs = static_tuple.as_tuples(node_refs)
                passed = static_tuple.as_tuples(keys[key])
                if node_refs != passed[1]:
                    details = '{} {} {}'.format(key, (value, node_refs), passed)
                    if self._inconsistency_fatal:
                        raise knit.KnitCorrupt(self, 'inconsistent details in add_records: %s' % details)
                    else:
                        trace.warning('inconsistent details in skipped record: %s', details)
                del keys[key]
                changed = True
        if changed:
            result = []
            if self._parents:
                for key, (value, node_refs) in keys.items():
                    result.append((key, value, node_refs))
            else:
                for key, (value, node_refs) in keys.items():
                    result.append((key, value))
            records = result
        key_dependencies = self._key_dependencies
        if key_dependencies is not None:
            if self._parents:
                for key, value, refs in records:
                    parents = refs[0]
                    key_dependencies.add_references(key, parents)
            else:
                for key, value, refs in records:
                    new_keys.add_key(key)
        self._add_callback(records)

    def _check_read(self):
        """Raise an exception if reads are not permitted."""
        if not self._is_locked():
            raise errors.ObjectNotLocked(self)

    def _check_write_ok(self):
        """Raise an exception if writes are not permitted."""
        if not self._is_locked():
            raise errors.ObjectNotLocked(self)

    def _get_entries(self, keys, check_present=False):
        """Get the entries for keys.

        Note: Callers are responsible for checking that the index is locked
        before calling this method.

        :param keys: An iterable of index key tuples.
        """
        keys = set(keys)
        found_keys = set()
        if self._parents:
            for node in self._graph_index.iter_entries(keys):
                yield node
                found_keys.add(node[1])
        else:
            for node in self._graph_index.iter_entries(keys):
                yield (node[0], node[1], node[2], ())
                found_keys.add(node[1])
        if check_present:
            missing_keys = keys.difference(found_keys)
            if missing_keys:
                raise errors.RevisionNotPresent(missing_keys.pop(), self)

    def find_ancestry(self, keys):
        """See CombinedGraphIndex.find_ancestry"""
        return self._graph_index.find_ancestry(keys, 0)

    def get_parent_map(self, keys):
        """Get a map of the parents of keys.

        :param keys: The keys to look up parents for.
        :return: A mapping from keys to parents. Absent keys are absent from
            the mapping.
        """
        self._check_read()
        nodes = self._get_entries(keys)
        result = {}
        if self._parents:
            for node in nodes:
                result[node[1]] = node[3][0]
        else:
            for node in nodes:
                result[node[1]] = None
        return result

    def get_missing_parents(self):
        """Return the keys of missing parents."""
        self._key_dependencies.satisfy_refs_for_keys(self.get_parent_map(self._key_dependencies.get_unsatisfied_refs()))
        return frozenset(self._key_dependencies.get_unsatisfied_refs())

    def get_build_details(self, keys):
        """Get the various build details for keys.

        Ghosts are omitted from the result.

        :param keys: An iterable of keys.
        :return: A dict of key:
            (index_memo, compression_parent, parents, record_details).

            * index_memo: opaque structure to pass to read_records to extract
              the raw data
            * compression_parent: Content that this record is built upon, may
              be None
            * parents: Logical parents of this node
            * record_details: extra information about the content which needs
              to be passed to Factory.parse_record
        """
        self._check_read()
        result = {}
        entries = self._get_entries(keys)
        for entry in entries:
            key = entry[1]
            if not self._parents:
                parents = None
            else:
                parents = entry[3][0]
            details = _GCBuildDetails(parents, self._node_to_position(entry))
            result[key] = details
        return result

    def keys(self):
        """Get all the keys in the collection.

        The keys are not ordered.
        """
        self._check_read()
        return [node[1] for node in self._graph_index.iter_all_entries()]

    def _node_to_position(self, node):
        """Convert an index value to position details."""
        bits = node[2].split(b' ')
        start = int(bits[0])
        start = self._int_cache.setdefault(start, start)
        stop = int(bits[1])
        stop = self._int_cache.setdefault(stop, stop)
        basis_end = int(bits[2])
        delta_end = int(bits[3])
        return (node[0], start, stop, basis_end, delta_end)

    def scan_unvalidated_index(self, graph_index):
        """Inform this _GCGraphIndex that there is an unvalidated index.

        This allows this _GCGraphIndex to keep track of any missing
        compression parents we may want to have filled in to make those
        indices valid.  It also allows _GCGraphIndex to track any new keys.

        :param graph_index: A GraphIndex
        """
        key_dependencies = self._key_dependencies
        if key_dependencies is None:
            return
        for node in graph_index.iter_all_entries():
            key_dependencies.add_references(node[1], node[3][0])