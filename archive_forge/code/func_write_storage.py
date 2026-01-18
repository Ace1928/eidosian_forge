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