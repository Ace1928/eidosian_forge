from webencodings import ascii_lower
from .serializer import _serialize_to, serialize_identifier, serialize_name
class ParenthesesBlock(Node):
    """A :diagram:`()-block`.

    .. code-block:: text

        '(' <content> ')'

    .. autoattribute:: type

    .. attribute:: content

        The content of the block, as list of :term:`component values`.
        The ``(`` and ``)`` markers themselves are not represented in the list.

    """
    __slots__ = ['content']
    type = '() block'
    repr_format = '<{self.__class__.__name__} ( … )>'

    def __init__(self, line, column, content):
        Node.__init__(self, line, column)
        self.content = content

    def _serialize_to(self, write):
        write('(')
        _serialize_to(self.content, write)
        write(')')