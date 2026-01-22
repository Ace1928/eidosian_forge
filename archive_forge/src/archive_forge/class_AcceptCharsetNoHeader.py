from collections import namedtuple
import re
import textwrap
import warnings
class AcceptCharsetNoHeader(_AcceptCharsetInvalidOrNoHeader):
    """
    Represent when there is no ``Accept-Charset`` header in the request.

    This object should not be modified. To add to the header, we can use the
    addition operators (``+`` and ``+=``), which return a new object (see the
    docstring for :meth:`AcceptCharsetNoHeader.__add__`).
    """

    @property
    def header_value(self):
        """
        (``str`` or ``None``) The header value.

        As there is no header in the request, this is ``None``.
        """
        return self._header_value

    @property
    def parsed(self):
        """
        (``list`` or ``None``) Parsed form of the header.

        As there is no header in the request, this is ``None``.
        """
        return self._parsed

    def __init__(self):
        """
        Create an :class:`AcceptCharsetNoHeader` instance.
        """
        self._header_value = None
        self._parsed = None
        self._parsed_nonzero = None

    def copy(self):
        """
        Create a copy of the header object.

        """
        return self.__class__()

    def __add__(self, other):
        """
        Add to header, creating a new header object.

        `other` can be:

        * ``None``
        * a ``str`` header value
        * a ``dict``, where keys are charsets and values are qvalues
        * a ``tuple`` or ``list``, where each item is a charset ``str`` or a
          ``tuple`` or ``list`` (charset, qvalue) pair (``str``'s and pairs
          can be mixed within the ``tuple`` or ``list``)
        * an :class:`AcceptCharsetValidHeader`, :class:`AcceptCharsetNoHeader`,
          or :class:`AcceptCharsetInvalidHeader` instance
        * object of any other type that returns a value for ``__str__``

        If `other` is a valid header value or an
        :class:`AcceptCharsetValidHeader` instance, a new
        :class:`AcceptCharsetValidHeader` instance with the valid header value
        is returned.

        If `other` is ``None``, an :class:`AcceptCharsetNoHeader` instance, an
        invalid header value, or an :class:`AcceptCharsetInvalidHeader`
        instance, a new :class:`AcceptCharsetNoHeader` instance is returned.
        """
        if isinstance(other, AcceptCharsetValidHeader):
            return AcceptCharsetValidHeader(header_value=other.header_value)
        if isinstance(other, (AcceptCharsetNoHeader, AcceptCharsetInvalidHeader)):
            return self.__class__()
        return self._add_instance_and_non_accept_charset_type(instance=self, other=other)

    def __radd__(self, other):
        """
        Add to header, creating a new header object.

        See the docstring for :meth:`AcceptCharsetNoHeader.__add__`.
        """
        return self.__add__(other=other)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def __str__(self):
        """Return the ``str`` ``'<no header in request>'``."""
        return '<no header in request>'

    def _add_instance_and_non_accept_charset_type(self, instance, other):
        if not other:
            return self.__class__()
        other_header_value = self._python_value_to_header_str(value=other)
        try:
            return AcceptCharsetValidHeader(header_value=other_header_value)
        except ValueError:
            return self.__class__()