from enum import IntEnum
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
class NgramPart:

    def __init__(self, nid: int):
        self.id_ = nid
        self._leafs_ = None

    def init(self):
        self._leafs_ = IntMap()

    def __repr__(self):
        if self.empty():
            return f'NgramPart({self.id_})'
        return f'NgramPart({self.id_}, {self.leafs_!r})'

    def empty(self):
        return self._leafs_ is None

    def has_leaves(self):
        return self._leafs_ is not None and len(self._leafs_) > 0

    @property
    def leafs_(self):
        if self._leafs_ is None:
            raise RuntimeError('NgramPart was not initialized.')
        return self._leafs_

    def find(self, key):
        if not self.has_leaves():
            return None
        if key in self._leafs_:
            return key
        return None

    def emplace(self, key, value):
        return self.leafs_.emplace(key, value)

    def __getitem__(self, key):
        return self._leafs_[key]