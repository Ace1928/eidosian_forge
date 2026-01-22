from webencodings import ascii_lower
from .serializer import _serialize_to, serialize_identifier, serialize_name
class StringToken(Node):
    """A :diagram:`string-token`.

    .. code-block:: text

        '"' <value> '"'

    .. autoattribute:: type

    .. attribute:: value

        The unescaped value, as a Unicode string, without the quotes.

    """
    __slots__ = ['value', 'representation']
    type = 'string'
    repr_format = '<{self.__class__.__name__} {self.representation}>'

    def __init__(self, line, column, value, representation):
        Node.__init__(self, line, column)
        self.value = value
        self.representation = representation

    def _serialize_to(self, write):
        write(self.representation)