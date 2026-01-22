import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config
def _di_subroutine_type(self, line, function, argmap):
    llfunc = function
    md = []
    for idx, llarg in enumerate(llfunc.args):
        if not llarg.name.startswith('arg.'):
            name = llarg.name.replace('.', '$')
            lltype = llarg.type
            size = self.cgctx.get_abi_sizeof(lltype)
            mdtype = self._var_type(lltype, size, datamodel=None)
            md.append(mdtype)
    for idx, (name, nbtype) in enumerate(argmap.items()):
        name = name.replace('.', '$')
        datamodel = self.cgctx.data_model_manager[nbtype]
        lltype = self.cgctx.get_value_type(nbtype)
        size = self.cgctx.get_abi_sizeof(lltype)
        mdtype = self._var_type(lltype, size, datamodel=datamodel)
        md.append(mdtype)
    return self.module.add_debug_info('DISubroutineType', {'types': self.module.add_metadata(md)})