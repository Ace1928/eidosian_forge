from webencodings import ascii_lower
from .serializer import _serialize_to, serialize_identifier, serialize_name
class NumberToken(Node):
    """A :diagram:`number-token`.

    .. autoattribute:: type

    .. attribute:: value

        The numeric value as a :class:`float`.

    .. attribute:: int_value

        The numeric value as an :class:`int`
        if :attr:`is_integer` is true, :obj:`None` otherwise.

    .. attribute:: is_integer

        Whether the token was syntactically an integer, as a boolean.

    .. attribute:: representation

        The CSS representation of the value, as a Unicode string.

    """
    __slots__ = ['value', 'int_value', 'is_integer', 'representation']
    type = 'number'
    repr_format = '<{self.__class__.__name__} {self.representation}>'

    def __init__(self, line, column, value, int_value, representation):
        Node.__init__(self, line, column)
        self.value = value
        self.int_value = int_value
        self.is_integer = int_value is not None
        self.representation = representation

    def _serialize_to(self, write):
        write(self.representation)