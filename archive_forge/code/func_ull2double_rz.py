import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def ull2double_rz(arg0, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0], {(core.dtype('uint64'),): ('__nv_ull2double_rz', core.dtype('fp64'))}, is_pure=True, _builder=_builder)