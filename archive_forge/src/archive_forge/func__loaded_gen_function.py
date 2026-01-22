import sys, os
import types
from . import model
from .error import VerificationError
def _loaded_gen_function(self, tp, name, module, library):
    assert isinstance(tp, model.FunctionPtrType)
    if tp.ellipsis:
        newfunction = self._load_constant(False, tp, name, module)
    else:
        indirections = []
        base_tp = tp
        if any((isinstance(typ, model.StructOrUnion) for typ in tp.args)) or isinstance(tp.result, model.StructOrUnion):
            indirect_args = []
            for i, typ in enumerate(tp.args):
                if isinstance(typ, model.StructOrUnion):
                    typ = model.PointerType(typ)
                    indirections.append((i, typ))
                indirect_args.append(typ)
            indirect_result = tp.result
            if isinstance(indirect_result, model.StructOrUnion):
                if indirect_result.fldtypes is None:
                    raise TypeError("'%s' is used as result type, but is opaque" % (indirect_result._get_c_name(),))
                indirect_result = model.PointerType(indirect_result)
                indirect_args.insert(0, indirect_result)
                indirections.insert(0, ('result', indirect_result))
                indirect_result = model.void_type
            tp = model.FunctionPtrType(tuple(indirect_args), indirect_result, tp.ellipsis)
        BFunc = self.ffi._get_cached_btype(tp)
        wrappername = '_cffi_f_%s' % name
        newfunction = module.load_function(BFunc, wrappername)
        for i, typ in indirections:
            newfunction = self._make_struct_wrapper(newfunction, i, typ, base_tp)
    setattr(library, name, newfunction)
    type(library)._cffi_dir.append(name)