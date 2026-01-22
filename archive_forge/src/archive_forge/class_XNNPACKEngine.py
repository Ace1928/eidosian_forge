import sys
import types
import torch
class XNNPACKEngine(types.ModuleType):

    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)
    enabled = _XNNPACKEnabled()