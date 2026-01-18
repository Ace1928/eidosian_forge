from llvmlite import ir
def visit_Instruction(self, instr):
    if instr.type == ir.IntType(64):
        if instr.opname in ['srem', 'urem', 'sdiv', 'udiv']:
            name = 'numba_{op}'.format(op=instr.opname)
            fn = self.module.globals.get(name)
            if fn is None:
                opty = instr.type
                sdivfnty = ir.FunctionType(opty, [opty, opty])
                fn = ir.Function(self.module, sdivfnty, name=name)
            repl = ir.CallInstr(parent=instr.parent, func=fn, args=instr.operands, name=instr.name)
            instr.parent.replace(instr, repl)