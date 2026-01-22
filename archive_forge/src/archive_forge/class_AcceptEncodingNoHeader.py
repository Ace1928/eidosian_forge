from collections import namedtuple
import re
import textwrap
import warnings
class AcceptEncodingNoHeader(_AcceptEncodingInvalidOrNoHeader):
    """
    Represent when there is no ``Accept-Encoding`` header in the request.

    This object should not be modified. To add to the header, we can use the
    addition operators (``+`` and ``+=``), which return a new object (see the
    docstring for :meth:`AcceptEncodingNoHeader.__add__`).
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
        Create an :class:`AcceptEncodingNoHeader` instance.
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
        * a ``dict``, with content-coding, ``identity`` or ``*`` ``str``'s as
          keys, and qvalue ``float``'s as values
        * a ``tuple`` or ``list``, where each item is either a header element
          ``str``, or a (content-coding/``identity``/``*``, qvalue) ``tuple``
          or ``list``
        * an :class:`AcceptEncodingValidHeader`,
          :class:`AcceptEncodingNoHeader`, or
          :class:`AcceptEncodingInvalidHeader` instance
        * object of any other type that returns a value for ``__str__``

        If `other` is a valid header value or an
        :class:`AcceptEncodingValidHeader` instance, a new
        :class:`AcceptEncodingValidHeader` instance with the valid header value
        is returned.

        If `other` is ``None``, an :class:`AcceptEncodingNoHeader` instance, an
        invalid header value, or an :class:`AcceptEncodingInvalidHeader`
        instance, a new :class:`AcceptEncodingNoHeader` instance is returned.
        """
        if isinstance(other, AcceptEncodingValidHeader):
            return AcceptEncodingValidHeader(header_value=other.header_value)
        if isinstance(other, (AcceptEncodingNoHeader, AcceptEncodingInvalidHeader)):
            return self.__class__()
        return self._add_instance_and_non_accept_encoding_type(instance=self, other=other)

    def __radd__(self, other):
        """
        Add to header, creating a new header object.

        See the docstring for :meth:`AcceptEncodingNoHeader.__add__`.
        """
        return self.__add__(other=other)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def __str__(self):
        """Return the ``str`` ``'<no header in request>'``."""
        return '<no header in request>'

    def _add_instance_and_non_accept_encoding_type(self, instance, other):
        if other is None:
            return self.__class__()
        other_header_value = self._python_value_to_header_str(value=other)
        try:
            return AcceptEncodingValidHeader(header_value=other_header_value)
        except ValueError:
            return self.__class__()