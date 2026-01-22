from llvmlite import ir
from llvmlite.ir.transforms import Visitor, CallVisitor
class FastFloatCallVisitor(CallVisitor):
    """
    A pass to change all float function calls to use fastmath.
    """

    def __init__(self, flags):
        self.flags = flags

    def visit_Call(self, instr):
        if instr.type in (ir.FloatType(), ir.DoubleType()):
            for flag in self.flags:
                instr.fastmath.add(flag)