import binascii
from collections import defaultdict, deque
from contextlib import suppress
from io import BytesIO, UnsupportedOperation
import os
import struct
import sys
from itertools import chain
from typing import (
import warnings
import zlib
from hashlib import sha1
from os import SEEK_CUR, SEEK_END
from struct import unpack_from
from .errors import ApplyDeltaError, ChecksumMismatch
from .file import GitFile
from .lru_cache import LRUSizeCache
from .objects import ObjectID, ShaFile, hex_to_sha, object_header, sha_to_hex
class DeltaChainIterator(Generic[T]):
    """Abstract iterator over pack data based on delta chains.

    Each object in the pack is guaranteed to be inflated exactly once,
    regardless of how many objects reference it as a delta base. As a result,
    memory usage is proportional to the length of the longest delta chain.

    Subclasses can override _result to define the result type of the iterator.
    By default, results are UnpackedObjects with the following members set:

    * offset
    * obj_type_num
    * obj_chunks
    * pack_type_num
    * delta_base     (for delta types)
    * comp_chunks    (if _include_comp is True)
    * decomp_chunks
    * decomp_len
    * crc32          (if _compute_crc32 is True)
    """
    _compute_crc32 = False
    _include_comp = False

    def __init__(self, file_obj, *, resolve_ext_ref=None) -> None:
        self._file = file_obj
        self._resolve_ext_ref = resolve_ext_ref
        self._pending_ofs: Dict[int, List[int]] = defaultdict(list)
        self._pending_ref: Dict[bytes, List[int]] = defaultdict(list)
        self._full_ofs: List[Tuple[int, int]] = []
        self._ext_refs: List[bytes] = []

    @classmethod
    def for_pack_data(cls, pack_data: PackData, resolve_ext_ref=None):
        walker = cls(None, resolve_ext_ref=resolve_ext_ref)
        walker.set_pack_data(pack_data)
        for unpacked in pack_data.iter_unpacked(include_comp=False):
            walker.record(unpacked)
        return walker

    @classmethod
    def for_pack_subset(cls, pack: 'Pack', shas: Iterable[bytes], *, allow_missing: bool=False, resolve_ext_ref=None):
        walker = cls(None, resolve_ext_ref=resolve_ext_ref)
        walker.set_pack_data(pack.data)
        todo = set()
        for sha in shas:
            assert isinstance(sha, bytes)
            try:
                off = pack.index.object_offset(sha)
            except KeyError:
                if not allow_missing:
                    raise
            else:
                todo.add(off)
        done = set()
        while todo:
            off = todo.pop()
            unpacked = pack.data.get_unpacked_object_at(off)
            walker.record(unpacked)
            done.add(off)
            base_ofs = None
            if unpacked.pack_type_num == OFS_DELTA:
                base_ofs = unpacked.offset - unpacked.delta_base
            elif unpacked.pack_type_num == REF_DELTA:
                with suppress(KeyError):
                    assert isinstance(unpacked.delta_base, bytes)
                    base_ofs = pack.index.object_index(unpacked.delta_base)
            if base_ofs is not None and base_ofs not in done:
                todo.add(base_ofs)
        return walker

    def record(self, unpacked: UnpackedObject) -> None:
        type_num = unpacked.pack_type_num
        offset = unpacked.offset
        if type_num == OFS_DELTA:
            base_offset = offset - unpacked.delta_base
            self._pending_ofs[base_offset].append(offset)
        elif type_num == REF_DELTA:
            assert isinstance(unpacked.delta_base, bytes)
            self._pending_ref[unpacked.delta_base].append(offset)
        else:
            self._full_ofs.append((offset, type_num))

    def set_pack_data(self, pack_data: PackData) -> None:
        self._file = pack_data._file

    def _walk_all_chains(self):
        for offset, type_num in self._full_ofs:
            yield from self._follow_chain(offset, type_num, None)
        yield from self._walk_ref_chains()
        assert not self._pending_ofs, repr(self._pending_ofs)

    def _ensure_no_pending(self) -> None:
        if self._pending_ref:
            raise UnresolvedDeltas([sha_to_hex(s) for s in self._pending_ref])

    def _walk_ref_chains(self):
        if not self._resolve_ext_ref:
            self._ensure_no_pending()
            return
        for base_sha, pending in sorted(self._pending_ref.items()):
            if base_sha not in self._pending_ref:
                continue
            try:
                type_num, chunks = self._resolve_ext_ref(base_sha)
            except KeyError:
                continue
            self._ext_refs.append(base_sha)
            self._pending_ref.pop(base_sha)
            for new_offset in pending:
                yield from self._follow_chain(new_offset, type_num, chunks)
        self._ensure_no_pending()

    def _result(self, unpacked: UnpackedObject) -> T:
        raise NotImplementedError

    def _resolve_object(self, offset: int, obj_type_num: int, base_chunks: List[bytes]) -> UnpackedObject:
        self._file.seek(offset)
        unpacked, _ = unpack_object(self._file.read, include_comp=self._include_comp, compute_crc32=self._compute_crc32)
        unpacked.offset = offset
        if base_chunks is None:
            assert unpacked.pack_type_num == obj_type_num
        else:
            assert unpacked.pack_type_num in DELTA_TYPES
            unpacked.obj_type_num = obj_type_num
            unpacked.obj_chunks = apply_delta(base_chunks, unpacked.decomp_chunks)
        return unpacked

    def _follow_chain(self, offset: int, obj_type_num: int, base_chunks: List[bytes]):
        todo = [(offset, obj_type_num, base_chunks)]
        while todo:
            offset, obj_type_num, base_chunks = todo.pop()
            unpacked = self._resolve_object(offset, obj_type_num, base_chunks)
            yield self._result(unpacked)
            unblocked = chain(self._pending_ofs.pop(unpacked.offset, []), self._pending_ref.pop(unpacked.sha(), []))
            todo.extend(((new_offset, unpacked.obj_type_num, unpacked.obj_chunks) for new_offset in unblocked))

    def __iter__(self) -> Iterator[T]:
        return self._walk_all_chains()

    def ext_refs(self):
        return self._ext_refs