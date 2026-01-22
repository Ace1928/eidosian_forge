import sys
import types
import torch
class VFModule(types.ModuleType):
    vf: types.ModuleType

    def __init__(self, name):
        super().__init__(name)
        self.vf = torch._C._VariableFunctions

    def __getattr__(self, attr):
        return getattr(self.vf, attr)