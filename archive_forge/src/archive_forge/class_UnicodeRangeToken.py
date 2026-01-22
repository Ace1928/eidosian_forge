from webencodings import ascii_lower
from .serializer import _serialize_to, serialize_identifier, serialize_name
class UnicodeRangeToken(Node):
    """A `unicode-range token <https://www.w3.org/TR/css-syntax-3/#urange>`_.

    .. autoattribute:: type

    .. attribute:: start

        The start of the range, as an integer between 0 and 1114111.

    .. attribute:: end

        The end of the range, as an integer between 0 and 1114111.
        Same as :attr:`start` if the source only specified one value.

    """
    __slots__ = ['start', 'end']
    type = 'unicode-range'
    repr_format = '<{self.__class__.__name__} {self.start} {self.end}>'

    def __init__(self, line, column, start, end):
        Node.__init__(self, line, column)
        self.start = start
        self.end = end

    def _serialize_to(self, write):
        if self.end == self.start:
            write('U+%X' % self.start)
        else:
            write('U+%X-%X' % (self.start, self.end))