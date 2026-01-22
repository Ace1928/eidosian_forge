from gitdb.db.base import (
from gitdb.util import LazyMixin
from gitdb.exc import (
from gitdb.pack import PackEntity
from functools import reduce
import os
import glob
class PackedDB(FileDBBase, ObjectDBR, CachingDB, LazyMixin):
    """A database operating on a set of object packs"""
    _sort_interval = 500

    def __init__(self, root_path):
        super().__init__(root_path)
        self._hit_count = 0
        self._st_mtime = 0

    def _set_cache_(self, attr):
        if attr == '_entities':
            self._entities = list()
            self.update_cache(force=True)

    def _sort_entities(self):
        self._entities.sort(key=lambda l: l[0], reverse=True)

    def _pack_info(self, sha):
        """:return: tuple(entity, index) for an item at the given sha
        :param sha: 20 or 40 byte sha
        :raise BadObject:
        **Note:** This method is not thread-safe, but may be hit in multi-threaded
            operation. The worst thing that can happen though is a counter that
            was not incremented, or the list being in wrong order. So we safe
            the time for locking here, lets see how that goes"""
        if self._hit_count % self._sort_interval == 0:
            self._sort_entities()
        for item in self._entities:
            index = item[2](sha)
            if index is not None:
                item[0] += 1
                self._hit_count += 1
                return (item[1], index)
        raise BadObject(sha)

    def has_object(self, sha):
        try:
            self._pack_info(sha)
            return True
        except BadObject:
            return False

    def info(self, sha):
        entity, index = self._pack_info(sha)
        return entity.info_at_index(index)

    def stream(self, sha):
        entity, index = self._pack_info(sha)
        return entity.stream_at_index(index)

    def sha_iter(self):
        for entity in self.entities():
            index = entity.index()
            sha_by_index = index.sha
            for index in range(index.size()):
                yield sha_by_index(index)

    def size(self):
        sizes = [item[1].index().size() for item in self._entities]
        return reduce(lambda x, y: x + y, sizes, 0)

    def store(self, istream):
        """Storing individual objects is not feasible as a pack is designed to
        hold multiple objects. Writing or rewriting packs for single objects is
        inefficient"""
        raise UnsupportedOperation()

    def update_cache(self, force=False):
        """
        Update our cache with the actually existing packs on disk. Add new ones,
        and remove deleted ones. We keep the unchanged ones

        :param force: If True, the cache will be updated even though the directory
            does not appear to have changed according to its modification timestamp.
        :return: True if the packs have been updated so there is new information,
            False if there was no change to the pack database"""
        stat = os.stat(self.root_path())
        if not force and stat.st_mtime <= self._st_mtime:
            return False
        self._st_mtime = stat.st_mtime
        pack_files = set(glob.glob(os.path.join(self.root_path(), 'pack-*.pack')))
        our_pack_files = {item[1].pack().path() for item in self._entities}
        for pack_file in pack_files - our_pack_files:
            entity = PackEntity(pack_file)
            self._entities.append([entity.pack().size(), entity, entity.index().sha_to_index])
        for pack_file in our_pack_files - pack_files:
            del_index = -1
            for i, item in enumerate(self._entities):
                if item[1].pack().path() == pack_file:
                    del_index = i
                    break
            assert del_index != -1
            del self._entities[del_index]
        self._sort_entities()
        return True

    def entities(self):
        """:return: list of pack entities operated upon by this database"""
        return [item[1] for item in self._entities]

    def partial_to_complete_sha(self, partial_binsha, canonical_length):
        """:return: 20 byte sha as inferred by the given partial binary sha
        :param partial_binsha: binary sha with less than 20 bytes
        :param canonical_length: length of the corresponding canonical representation.
            It is required as binary sha's cannot display whether the original hex sha
            had an odd or even number of characters
        :raise AmbiguousObjectName:
        :raise BadObject: """
        candidate = None
        for item in self._entities:
            item_index = item[1].index().partial_sha_to_index(partial_binsha, canonical_length)
            if item_index is not None:
                sha = item[1].index().sha(item_index)
                if candidate and candidate != sha:
                    raise AmbiguousObjectName(partial_binsha)
                candidate = sha
        if candidate:
            return candidate
        raise BadObject(partial_binsha)