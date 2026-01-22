from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class GEPInstr(Instruction):

    def __init__(self, parent, ptr, indices, inbounds, name):
        typ = ptr.type
        lasttyp = None
        lastaddrspace = 0
        for i in indices:
            lasttyp, typ = (typ, typ.gep(i))
            if isinstance(lasttyp, types.PointerType):
                lastaddrspace = lasttyp.addrspace
        if not isinstance(typ, types.PointerType) and isinstance(lasttyp, types.PointerType):
            typ = lasttyp
        else:
            typ = typ.as_pointer(lastaddrspace)
        super(GEPInstr, self).__init__(parent, typ, 'getelementptr', [ptr] + list(indices), name=name)
        self.pointer = ptr
        self.indices = indices
        self.inbounds = inbounds

    def descr(self, buf):
        indices = ['{0} {1}'.format(i.type, i.get_reference()) for i in self.indices]
        op = 'getelementptr inbounds' if self.inbounds else 'getelementptr'
        buf.append('{0} {1}, {2} {3}, {4} {5}\n'.format(op, self.pointer.type.pointee, self.pointer.type, self.pointer.get_reference(), ', '.join(indices), self._stringify_metadata(leading_comma=True)))