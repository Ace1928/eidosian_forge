import array
import contextlib
import enum
import struct
def Adder(self, type_):
    return {Type.BOOL: self.Bool, Type.INT: self.Int, Type.INDIRECT_INT: self.IndirectInt, Type.UINT: self.UInt, Type.INDIRECT_UINT: self.IndirectUInt, Type.FLOAT: self.Float, Type.INDIRECT_FLOAT: self.IndirectFloat, Type.KEY: self.Key, Type.BLOB: self.Blob, Type.STRING: self.String}[type_]