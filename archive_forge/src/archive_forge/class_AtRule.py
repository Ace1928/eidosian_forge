from webencodings import ascii_lower
from .serializer import _serialize_to, serialize_identifier, serialize_name
class AtRule(Node):
    """An :diagram:`at-rule`.

    .. code-block:: text

        @<at_keyword> <prelude> '{' <content> '}'
        @<at_keyword> <prelude> ';'

    The interpretation of at-rules depend on their at-keyword
    as well as their context.
    Most types of at-rules (ie. at-keyword values)
    are only allowed in some context,
    and must either end with a {} block or a semicolon.

    .. autoattribute:: type

    .. attribute:: at_keyword

        The unescaped value of the rule’s at-keyword,
        without the ``@`` symbol, as a Unicode string.

    .. attribute:: lower_at_keyword

        Same as :attr:`at_keyword` but normalized to *ASCII lower case*,
        see :func:`~webencodings.ascii_lower`.
        This is the value to use when comparing to a CSS at-keyword.

        .. code-block:: python

            if node.type == 'at-rule' and node.lower_at_keyword == 'import':

    .. attribute:: prelude

        The rule’s prelude, the part before the {} block or semicolon,
        as a list of :term:`component values`.

    .. attribute:: content

        The rule’s content, if any.
        The block’s content as a list of :term:`component values`
        for at-rules with a {} block,
        or :obj:`None` for at-rules ending with a semicolon.

    """
    __slots__ = ['at_keyword', 'lower_at_keyword', 'prelude', 'content']
    type = 'at-rule'
    repr_format = '<{self.__class__.__name__} @{self.at_keyword} … {{ … }}>'

    def __init__(self, line, column, at_keyword, lower_at_keyword, prelude, content):
        Node.__init__(self, line, column)
        self.at_keyword = at_keyword
        self.lower_at_keyword = lower_at_keyword
        self.prelude = prelude
        self.content = content

    def _serialize_to(self, write):
        write('@')
        write(serialize_identifier(self.at_keyword))
        _serialize_to(self.prelude, write)
        if self.content is None:
            write(';')
        else:
            write('{')
            _serialize_to(self.content, write)
            write('}')