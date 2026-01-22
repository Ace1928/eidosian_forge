from llvmlite import ir
from llvmlite.ir.transforms import Visitor, CallVisitor

    Rewrite the given LLVM module to use fastmath everywhere.
    