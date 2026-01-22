import sys
import types
import torch
class _XNNPACKEnabled:

    def __get__(self, obj, objtype):
        return torch._C._is_xnnpack_enabled()

    def __set__(self, obj, val):
        raise RuntimeError('Assignment not supported')