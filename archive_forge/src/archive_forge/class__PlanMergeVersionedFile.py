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
class _PlanMergeVersionedFile(VersionedFiles):
    """A VersionedFile for uncommitted and committed texts.

    It is intended to allow merges to be planned with working tree texts.
    It implements only the small part of the VersionedFiles interface used by
    PlanMerge.  It falls back to multiple versionedfiles for data not stored in
    _PlanMergeVersionedFile itself.

    :ivar: fallback_versionedfiles a list of VersionedFiles objects that can be
        queried for missing texts.
    """

    def __init__(self, file_id):
        """Create a _PlanMergeVersionedFile.

        :param file_id: Used with _PlanMerge code which is not yet fully
            tuple-keyspace aware.
        """
        self._file_id = file_id
        self.fallback_versionedfiles = []
        self._parents = {}
        self._lines = {}
        self._providers = [_mod_graph.DictParentsProvider(self._parents)]

    def plan_merge(self, ver_a, ver_b, base=None):
        """See VersionedFile.plan_merge"""
        from ..merge import _PlanMerge
        if base is None:
            return _PlanMerge(ver_a, ver_b, self, (self._file_id,)).plan_merge()
        old_plan = list(_PlanMerge(ver_a, base, self, (self._file_id,)).plan_merge())
        new_plan = list(_PlanMerge(ver_a, ver_b, self, (self._file_id,)).plan_merge())
        return _PlanMerge._subtract_plans(old_plan, new_plan)

    def plan_lca_merge(self, ver_a, ver_b, base=None):
        from ..merge import _PlanLCAMerge
        graph = _mod_graph.Graph(self)
        new_plan = _PlanLCAMerge(ver_a, ver_b, self, (self._file_id,), graph).plan_merge()
        if base is None:
            return new_plan
        old_plan = _PlanLCAMerge(ver_a, base, self, (self._file_id,), graph).plan_merge()
        return _PlanLCAMerge._subtract_plans(list(old_plan), list(new_plan))

    def add_content(self, factory):
        return self.add_lines(factory.key, factory.parents, factory.get_bytes_as('lines'))

    def add_lines(self, key, parents, lines):
        """See VersionedFiles.add_lines

        Lines are added locally, not to fallback versionedfiles.  Also, ghosts
        are permitted.  Only reserved ids are permitted.
        """
        if not isinstance(key, tuple):
            raise TypeError(key)
        if not revision.is_reserved_id(key[-1]):
            raise ValueError('Only reserved ids may be used')
        if parents is None:
            raise ValueError('Parents may not be None')
        if lines is None:
            raise ValueError('Lines may not be None')
        self._parents[key] = tuple(parents)
        self._lines[key] = lines

    def get_record_stream(self, keys, ordering, include_delta_closure):
        pending = set(keys)
        for key in keys:
            if key in self._lines:
                lines = self._lines[key]
                parents = self._parents[key]
                pending.remove(key)
                yield ChunkedContentFactory(key, parents, None, lines, chunks_are_lines=True)
        for versionedfile in self.fallback_versionedfiles:
            for record in versionedfile.get_record_stream(pending, 'unordered', True):
                if record.storage_kind == 'absent':
                    continue
                else:
                    pending.remove(record.key)
                    yield record
            if not pending:
                return
        for key in pending:
            yield AbsentContentFactory(key)

    def get_parent_map(self, keys):
        """See VersionedFiles.get_parent_map"""
        keys = set(keys)
        result = {}
        if revision.NULL_REVISION in keys:
            keys.remove(revision.NULL_REVISION)
            result[revision.NULL_REVISION] = ()
        self._providers = self._providers[:1] + self.fallback_versionedfiles
        result.update(_mod_graph.StackedParentsProvider(self._providers).get_parent_map(keys))
        for key, parents in result.items():
            if parents == ():
                result[key] = (revision.NULL_REVISION,)
        return result