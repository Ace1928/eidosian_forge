import ctypes
import functools
import hashlib
import os.path
import struct
from collections import defaultdict
from typing import Dict, Optional, Set
import torch
import torch._prims as prims
import torch._utils
import torch.nn.functional as F
from torch._C import default_generator
from torch.multiprocessing.reductions import StorageWeakRef
class ContentStoreWriter:

    def __init__(self, loc: str, stable_hash: bool=False) -> None:
        self.loc: str = loc
        self.seen_storage_hashes: Set[str] = set()
        self.stable_hash = stable_hash

    def write_storage(self, storage: torch.UntypedStorage) -> str:
        h = hash_storage(storage, stable_hash=self.stable_hash)
        if h in self.seen_storage_hashes:
            return h
        subfolder = os.path.join(self.loc, 'storages')
        os.makedirs(subfolder, exist_ok=True)
        target = os.path.join(subfolder, h)
        if os.path.exists(target):
            return h
        torch.save(storage, target)
        self.seen_storage_hashes.add(h)
        return h

    def compute_tensor_metadata(self, t: torch.Tensor, h=None):
        if h is None:
            h = hash_storage(t.untyped_storage(), stable_hash=self.stable_hash)
        return (t.dtype, h, t.storage_offset(), tuple(t.shape), t.stride(), torch._utils.get_tensor_metadata(t))

    def write_tensor(self, name: str, t: torch.Tensor) -> None:
        storage = t.untyped_storage()
        h = self.write_storage(storage)
        d, f = os.path.split(name)
        payload = self.compute_tensor_metadata(t, h=h)
        subfolder = os.path.join(self.loc, 'tensors', d)
        os.makedirs(subfolder, exist_ok=True)
        torch.save(payload, os.path.join(subfolder, f))