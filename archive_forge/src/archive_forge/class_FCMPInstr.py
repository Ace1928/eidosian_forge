from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class FCMPInstr(CompareInstr):
    OPNAME = 'fcmp'
    VALID_OP = {'false': 'no comparison, always returns false', 'oeq': 'ordered and equal', 'ogt': 'ordered and greater than', 'oge': 'ordered and greater than or equal', 'olt': 'ordered and less than', 'ole': 'ordered and less than or equal', 'one': 'ordered and not equal', 'ord': 'ordered (no nans)', 'ueq': 'unordered or equal', 'ugt': 'unordered or greater than', 'uge': 'unordered or greater than or equal', 'ult': 'unordered or less than', 'ule': 'unordered or less than or equal', 'une': 'unordered or not equal', 'uno': 'unordered (either nans)', 'true': 'no comparison, always returns true'}
    VALID_FLAG = {'nnan', 'ninf', 'nsz', 'arcp', 'contract', 'afn', 'reassoc', 'fast'}