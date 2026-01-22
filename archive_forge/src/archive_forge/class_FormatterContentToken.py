import operator
from debian._deb822_repro._util import BufferingIterator
from debian._deb822_repro.tokens import Deb822Token
class FormatterContentToken(object):
    """Typed, tagged text for use with the formatting API

    The FormatterContentToken is used by the formatting API and provides the
    formatter callback with context about the textual tokens it is supposed
    to format.
    """
    __slots__ = ('_text', '_content_type')

    def __init__(self, text, content_type):
        self._text = text
        self._content_type = content_type

    @classmethod
    def from_token_or_element(cls, token_or_element):
        if isinstance(token_or_element, Deb822Token):
            if token_or_element.is_comment:
                return cls.comment_token(token_or_element.text)
            if token_or_element.is_whitespace:
                raise ValueError('FormatterContentType cannot be whitespace')
            return cls.value_token(token_or_element.text)
        return cls.value_token(token_or_element.convert_to_text())

    @classmethod
    def separator_token(cls, text):
        if text == ' ':
            return SPACE_SEPARATOR_FT
        if text == ',':
            return COMMA_SEPARATOR_FT
        return cls(text, _CONTENT_TYPE_SEPARATOR)

    @classmethod
    def comment_token(cls, text):
        """Generates a single comment token with the provided text

        Mostly useful for creating test cases
        """
        return cls(text, _CONTENT_TYPE_COMMENT)

    @classmethod
    def value_token(cls, text):
        """Generates a single value token with the provided text

        Mostly useful for creating test cases
        """
        return cls(text, _CONTENT_TYPE_VALUE)

    @property
    def is_comment(self):
        """True if this formatter token represent a comment

        This should be used for determining whether the token is a comment
        or not. It might be tempting to check whether the text in the token
        starts with a "#" but that is insufficient because a value *can*
        start with that as well.  Whether it is a comment or a value is
        based on the context (it is a comment if and only if the "#" was
        at the start of a line) but the formatter often do not have the
        context available to assert this.

        The formatter *should* preserve the order of comments and interleave
        between the value tokens in the same order as it see them.  Failing
        to preserve the order of comments and values can cause confusing
        comments (such as associating the comment with a different value
        than it was written for).

        The formatter *may* discard comment tokens if it does not want to
        preserve them.  If so, they would be omitted in the output, which
        may be acceptable in some cases.  This is a lot better than
        re-ordering comments.

        Formatters must be aware of the following special cases for comments:
         * Comments *MUST* be emitted after a newline.  If the very first token
           is a comment, the formatter is expected to emit a newline before it
           as well (Fields cannot start immediately on a comment).
        """
        return self._content_type is _CONTENT_TYPE_COMMENT

    @property
    def is_value(self):
        """True if this formatter token represents a semantic value

        The formatter *MUST* preserve values as-in in its output.  It may
        "unpack" it from the token (as in, return it as a part of a plain
        str) but the value content must not be changed nor re-ordered relative
        to other value tokens (as that could change the meaning of the field).
        """
        return self._content_type is _CONTENT_TYPE_VALUE

    @property
    def is_separator(self):
        """True if this formatter token represents a separator token

        The formatter is not required to preserve the provided separators but it
        is required to properly separate values.  In fact, often is a lot easier
        to discard existing separator tokens.  As an example, in whitespace
        separated list of values space, tab and newline all counts as separator.
        However, formatting-wise, there is a world of difference between the
        a space, tab and a newline. In particularly, newlines must be followed
        by an additional space or tab (to act as a value continuation line) if
        there is a value following it (otherwise, the generated output is
        invalid).
        """
        return self._content_type is _CONTENT_TYPE_SEPARATOR

    @property
    def is_whitespace(self):
        """True if this formatter token represents a whitespace token"""
        return self._content_type is _CONTENT_TYPE_SEPARATOR and self._text.isspace()

    @property
    def text(self):
        """The actual context of the token

        This field *must not* be used to determine the type of token.  The
        formatter cannot reliably tell whether "#..." is a comment or a value
        (it can be both).  Use is_value and is_comment instead for discriminating
        token types.

        For value tokens, this the concrete value to be omitted.

        For comment token, this is the full comment text.

        This is the same as str(token).
        """
        return self._text

    def __str__(self):
        return self._text

    def __repr__(self):
        return '{}({!r}, {}=True)'.format(self.__class__.__name__, self._text, self._content_type)