from llvmlite.ir import CallInstr
def visit_BasicBlock(self, bb):
    self._basic_block = bb
    for instr in bb.instructions:
        self.visit_Instruction(instr)