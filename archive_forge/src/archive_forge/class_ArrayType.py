import struct
from llvmlite.ir._utils import _StrCaching
class ArrayType(Aggregate):
    """
    The type for fixed-size homogenous arrays (e.g. "[f32 x 3]").
    """

    def __init__(self, element, count):
        self.element = element
        self.count = count

    @property
    def elements(self):
        return _Repeat(self.element, self.count)

    def __len__(self):
        return self.count

    def _to_string(self):
        return '[%d x %s]' % (self.count, self.element)

    def __eq__(self, other):
        if isinstance(other, ArrayType):
            return self.element == other.element and self.count == other.count

    def __hash__(self):
        return hash(ArrayType)

    def gep(self, i):
        """
        Resolve the type of the i-th element (for getelementptr lookups).
        """
        if not isinstance(i.type, IntType):
            raise TypeError(i.type)
        return self.element

    def format_constant(self, value):
        itemstring = ', '.join(['{0} {1}'.format(x.type, x.get_reference()) for x in value])
        return '[{0}]'.format(itemstring)