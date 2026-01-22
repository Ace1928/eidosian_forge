import copy
import functools
import getpass
import itertools
import logging
import os
import subprocess
import tempfile
import textwrap
from collections import Counter
from importlib import import_module
from typing import Callable, Optional, TypeVar
import torch
import torch._prims_common as utils
import torch._subclasses.meta_utils
from torch._dynamo.testing import rand_strided
from torch._prims_common import is_float_dtype
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._content_store import ContentStoreReader, ContentStoreWriter
from . import config
from .utils import clone_inputs, get_debug_dir
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
class InputWriter:

    def __init__(self, save_dir, *, stable_hash=False):
        self._lines = []
        self.storage_counter = itertools.count()
        self.save_dir = save_dir
        self.store = ContentStoreWriter(save_dir, stable_hash=stable_hash) if save_dir is not None else None
        self.seen_storages = {}

    def lines(self):
        r = ['def load_args(reader):']
        r.extend((f'    {l}' for l in self._lines))
        r.append('load_args._version = 0')
        return r

    def storage(self, untyped_storage, *, dtype_hint=None, device_hint=None) -> str:
        ws = StorageWeakRef(untyped_storage)
        v = self.seen_storages.get(ws)
        if v is not None:
            return v
        v = f'buf{next(self.storage_counter)}'
        maybe_dtype_hint = ''
        if _dtype_or_default(None) != _dtype_or_default(dtype_hint):
            maybe_dtype_hint = f', dtype_hint={dtype_hint!r}'
        maybe_device = ''
        device = untyped_storage.device
        if device.type == 'meta':
            assert device_hint is not None
            device = device_hint
        if _device_or_default(None) != device:
            maybe_device = f', device={device!r}'
        nbytes = untyped_storage.nbytes()
        storage_hash = None
        if self.store is not None and untyped_storage.device.type != 'meta':
            storage_hash = self.store.write_storage(untyped_storage)
        self._lines.append(f'{v} = reader.storage({storage_hash!r}, {nbytes!r}{maybe_device}{maybe_dtype_hint})')
        self.seen_storages[ws] = v
        return v

    def tensor(self, name, t) -> None:
        storage = self.storage(t.untyped_storage(), dtype_hint=t.dtype, device_hint=t.device)
        args = []
        if _stride_or_default(None, shape=t.shape) != t.stride():
            args.append(str(tuple(t.stride())))
        if _dtype_or_default(None) != t.dtype:
            args.append(f'dtype={t.dtype!r}')
        if _storage_offset_or_default(None) != t.storage_offset():
            args.append(f'storage_offset={t.storage_offset()!r}')
        tensor_metadata = torch._utils.get_tensor_metadata(t)
        if tensor_metadata:
            args.extend((f'{k}={v!r}' for k, v in tensor_metadata.items()))
        if _requires_grad_or_default(None) != t.requires_grad:
            args.append(f'requires_grad={t.requires_grad!r}')
        is_leaf = torch._subclasses.meta_utils.safe_is_leaf(t)
        if _is_leaf_or_default(None) != is_leaf:
            args.append(f'is_leaf={is_leaf!r}')
        self._lines.append('reader.tensor(' + ', '.join([storage, str(tuple(t.shape)), *args]) + f')  # {name}')

    def symint(self, name, val) -> None:
        if isinstance(val, torch.SymInt):
            val = val.node.hint
        self._lines.append(f'reader.symint({val!r})  # {name}')