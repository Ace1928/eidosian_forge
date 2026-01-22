from __future__ import absolute_import
from .Visitor import CythonTransform
from .ModuleNode import ModuleNode
from .Errors import CompileError
from .UtilityCode import CythonUtilityCode
from .Code import UtilityCode, TempitaUtilityCode
from . import Options
from . import Interpreter
from . import PyrexTypes
from . import Naming
from . import Symtab
class BufferEntry(object):

    def __init__(self, entry):
        self.entry = entry
        self.type = entry.type
        self.cname = entry.buffer_aux.buflocal_nd_var.cname
        self.buf_ptr = '%s.rcbuffer->pybuffer.buf' % self.cname
        self.buf_ptr_type = entry.type.buffer_ptr_type
        self.init_attributes()

    def init_attributes(self):
        self.shape = self.get_buf_shapevars()
        self.strides = self.get_buf_stridevars()
        self.suboffsets = self.get_buf_suboffsetvars()

    def get_buf_suboffsetvars(self):
        return self._for_all_ndim('%s.diminfo[%d].suboffsets')

    def get_buf_stridevars(self):
        return self._for_all_ndim('%s.diminfo[%d].strides')

    def get_buf_shapevars(self):
        return self._for_all_ndim('%s.diminfo[%d].shape')

    def _for_all_ndim(self, s):
        return [s % (self.cname, i) for i in range(self.type.ndim)]

    def generate_buffer_lookup_code(self, code, index_cnames):
        params = []
        nd = self.type.ndim
        mode = self.type.mode
        if mode == 'full':
            for i, s, o in zip(index_cnames, self.get_buf_stridevars(), self.get_buf_suboffsetvars()):
                params.append(i)
                params.append(s)
                params.append(o)
            funcname = '__Pyx_BufPtrFull%dd' % nd
            funcgen = buf_lookup_full_code
        else:
            if mode == 'strided':
                funcname = '__Pyx_BufPtrStrided%dd' % nd
                funcgen = buf_lookup_strided_code
            elif mode == 'c':
                funcname = '__Pyx_BufPtrCContig%dd' % nd
                funcgen = buf_lookup_c_code
            elif mode == 'fortran':
                funcname = '__Pyx_BufPtrFortranContig%dd' % nd
                funcgen = buf_lookup_fortran_code
            else:
                assert False
            for i, s in zip(index_cnames, self.get_buf_stridevars()):
                params.append(i)
                params.append(s)
        if funcname not in code.globalstate.utility_codes:
            code.globalstate.utility_codes.add(funcname)
            protocode = code.globalstate['utility_code_proto']
            defcode = code.globalstate['utility_code_def']
            funcgen(protocode, defcode, name=funcname, nd=nd)
        buf_ptr_type_code = self.buf_ptr_type.empty_declaration_code()
        ptrcode = '%s(%s, %s, %s)' % (funcname, buf_ptr_type_code, self.buf_ptr, ', '.join(params))
        return ptrcode