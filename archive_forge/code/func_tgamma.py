import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def tgamma(arg0, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0], {(core.dtype('fp32'),): ('__nv_tgammaf', core.dtype('fp32')), (core.dtype('fp64'),): ('__nv_tgamma', core.dtype('fp64'))}, is_pure=True, _builder=_builder)