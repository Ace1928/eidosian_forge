import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
class ExitUnchanged(Exception):
    """Exception used to skip the peephole optimizer"""
    pass