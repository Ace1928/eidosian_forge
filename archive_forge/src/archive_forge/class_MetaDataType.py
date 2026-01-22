import struct
from llvmlite.ir._utils import _StrCaching
class MetaDataType(Type):

    def _to_string(self):
        return 'metadata'

    def as_pointer(self):
        raise TypeError

    def __eq__(self, other):
        return isinstance(other, MetaDataType)

    def __hash__(self):
        return hash(MetaDataType)