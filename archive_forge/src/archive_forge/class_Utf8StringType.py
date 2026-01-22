import base64
import inspect
import builtins
class Utf8StringType(TypeDescr):

    @staticmethod
    def encode(v):
        return str(v, 'utf-8')

    @staticmethod
    def decode(v):
        return v.encode('utf-8')