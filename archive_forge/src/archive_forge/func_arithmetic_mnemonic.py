from pyparsing import *
from sys import stdin, argv, exit
def arithmetic_mnemonic(self, op_name, op_type):
    """Generates an arithmetic instruction mnemonic"""
    return self.OPERATIONS[op_name] + self.OPSIGNS[op_type]