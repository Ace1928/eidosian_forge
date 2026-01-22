from llvmlite.ir import CallInstr
class CallVisitor(Visitor):

    def visit_Instruction(self, instr):
        if isinstance(instr, CallInstr):
            self.visit_Call(instr)

    def visit_Call(self, instr):
        raise NotImplementedError