import struct
from llvmlite.ir._utils import _StrCaching
class IdentifiedStructType(BaseStructType):
    """
    A type which is a named alias for another struct type, akin to a typedef.
    While literal struct types can be structurally equal (see
    LiteralStructType), identified struct types are compared by name.

    Do not use this directly.
    """
    null = 'zeroinitializer'

    def __init__(self, context, name, packed=False):
        """
        *context* is a llvmlite.ir.Context.
        *name* is the identifier for the new struct type.
        *packed* controls the use of packed layout.
        """
        assert name
        self.context = context
        self.name = name
        self.elements = None
        self.packed = packed

    def _to_string(self):
        return '%{name}'.format(name=_wrapname(self.name))

    def get_declaration(self):
        """
        Returns the string for the declaration of the type
        """
        if self.is_opaque:
            out = '{strrep} = type opaque'.format(strrep=str(self))
        else:
            out = '{strrep} = type {struct}'.format(strrep=str(self), struct=self.structure_repr())
        return out

    def __eq__(self, other):
        if isinstance(other, IdentifiedStructType):
            return self.name == other.name

    def __hash__(self):
        return hash(IdentifiedStructType)

    def set_body(self, *elems):
        if not self.is_opaque:
            raise RuntimeError('{name} is already defined'.format(name=self.name))
        self.elements = tuple(elems)