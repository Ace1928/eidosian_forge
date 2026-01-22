from thrift.Thrift import TType, TMessageType, TException
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol, TProtocol
class AnnotationType:
    BOOL = 0
    BYTES = 1
    I16 = 2
    I32 = 3
    I64 = 4
    DOUBLE = 5
    STRING = 6
    _VALUES_TO_NAMES = {0: 'BOOL', 1: 'BYTES', 2: 'I16', 3: 'I32', 4: 'I64', 5: 'DOUBLE', 6: 'STRING'}
    _NAMES_TO_VALUES = {'BOOL': 0, 'BYTES': 1, 'I16': 2, 'I32': 3, 'I64': 4, 'DOUBLE': 5, 'STRING': 6}