import base64
import inspect
import builtins
class AsciiStringType(TypeDescr):

    @staticmethod
    def encode(v):
        if isinstance(v, str):
            return v
        return str(v, 'ascii')

    @staticmethod
    def decode(v):
        return v