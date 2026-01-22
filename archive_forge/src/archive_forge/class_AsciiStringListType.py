import base64
import inspect
import builtins
class AsciiStringListType(TypeDescr):

    @staticmethod
    def encode(v):
        return [AsciiStringType.encode(x) for x in v]

    @staticmethod
    def decode(v):
        return [AsciiStringType.decode(x) for x in v]