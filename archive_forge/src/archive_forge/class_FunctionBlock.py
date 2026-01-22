from webencodings import ascii_lower
from .serializer import _serialize_to, serialize_identifier, serialize_name
class FunctionBlock(Node):
    """A :diagram:`function-block`.

    .. code-block:: text

        <name> '(' <arguments> ')'

    .. autoattribute:: type

    .. attribute:: name

        The unescaped name of the function, as a Unicode string.

    .. attribute:: lower_name

        Same as :attr:`name` but normalized to *ASCII lower case*,
        see :func:`~webencodings.ascii_lower`.
        This is the value to use when comparing to a CSS function name.

    .. attribute:: arguments

        The arguments of the function, as list of :term:`component values`.
        The ``(`` and ``)`` markers themselves are not represented in the list.
        Commas are not special, but represented as :obj:`LiteralToken` objects
        in the list.

    """
    __slots__ = ['name', 'lower_name', 'arguments']
    type = 'function'
    repr_format = '<{self.__class__.__name__} {self.name}( … )>'

    def __init__(self, line, column, name, arguments):
        Node.__init__(self, line, column)
        self.name = name
        self.lower_name = ascii_lower(name)
        self.arguments = arguments

    def _serialize_to(self, write):
        write(serialize_identifier(self.name))
        write('(')
        _serialize_to(self.arguments, write)
        function = self
        while isinstance(function, FunctionBlock) and function.arguments:
            eof_in_string = isinstance(function.arguments[-1], ParseError) and function.arguments[-1].kind == 'eof-in-string'
            if eof_in_string:
                return
            function = function.arguments[-1]
        write(')')