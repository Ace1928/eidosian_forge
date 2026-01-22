from webencodings import ascii_lower
from .serializer import _serialize_to, serialize_identifier, serialize_name
class DimensionToken(Node):
    """A :diagram:`dimension-token`.

    .. code-block:: text

        <representation> <unit>

    .. autoattribute:: type

    .. attribute:: value

        The value numeric as a :class:`float`.

    .. attribute:: int_value

        The numeric value as an :class:`int`
        if the token was syntactically an integer,
        or :obj:`None`.

    .. attribute:: is_integer

        Whether the token’s value was syntactically an integer, as a boolean.

    .. attribute:: representation

        The CSS representation of the value without the unit,
        as a Unicode string.

    .. attribute:: unit

        The unescaped unit, as a Unicode string.

    .. attribute:: lower_unit

        Same as :attr:`unit` but normalized to *ASCII lower case*,
        see :func:`~webencodings.ascii_lower`.
        This is the value to use when comparing to a CSS unit.

        .. code-block:: python

            if node.type == 'dimension' and node.lower_unit == 'px':

    """
    __slots__ = ['value', 'int_value', 'is_integer', 'representation', 'unit', 'lower_unit']
    type = 'dimension'
    repr_format = '<{self.__class__.__name__} {self.representation}{self.unit}>'

    def __init__(self, line, column, value, int_value, representation, unit):
        Node.__init__(self, line, column)
        self.value = value
        self.int_value = int_value
        self.is_integer = int_value is not None
        self.representation = representation
        self.unit = unit
        self.lower_unit = ascii_lower(unit)

    def _serialize_to(self, write):
        write(self.representation)
        unit = self.unit
        if unit in ('e', 'E') or unit.startswith(('e-', 'E-')):
            write('\\65 ')
            write(serialize_name(unit[1:]))
        else:
            write(serialize_identifier(unit))