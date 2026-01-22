from collections import namedtuple
import re
import textwrap
import warnings
class AcceptInvalidHeader(_AcceptInvalidOrNoHeader):
    """
    Represent an invalid ``Accept`` header.

    An invalid header is one that does not conform to
    :rfc:`7231#section-5.3.2`.

    :rfc:`7231` does not provide any guidance on what should happen if the
    ``Accept`` header has an invalid value. This implementation disregards the
    header, and treats it as if there is no ``Accept`` header in the request.

    This object should not be modified. To add to the header, we can use the
    addition operators (``+`` and ``+=``), which return a new object (see the
    docstring for :meth:`AcceptInvalidHeader.__add__`).
    """

    @property
    def header_value(self):
        """(``str`` or ``None``) The header value."""
        return self._header_value

    @property
    def parsed(self):
        """
        (``list`` or ``None``) Parsed form of the header.

        As the header is invalid and cannot be parsed, this is ``None``.
        """
        return self._parsed

    def __init__(self, header_value):
        """
        Create an :class:`AcceptInvalidHeader` instance.
        """
        self._header_value = header_value
        self._parsed = None
        self._parsed_nonzero = None

    def copy(self):
        """
        Create a copy of the header object.

        """
        return self.__class__(self._header_value)

    def __add__(self, other):
        """
        Add to header, creating a new header object.

        `other` can be:

        * ``None``
        * a ``str`` header value
        * a ``dict``, with media ranges ``str``'s (including any media type
          parameters) as keys, and either qvalues ``float``'s or (*qvalues*,
          *extension_params*) tuples as values, where *extension_params* is a
          ``str`` of the extension parameters segment of the header element,
          starting with the first '``;``'
        * a ``tuple`` or ``list``, where each item is either a header element
          ``str``, or a (*media_range*, *qvalue*, *extension_params*) ``tuple``
          or ``list`` where *media_range* is a ``str`` of the media range
          including any media type parameters, and *extension_params* is a
          ``str`` of the extension parameters segment of the header element,
          starting with the first '``;``'
        * an :class:`AcceptValidHeader`, :class:`AcceptNoHeader`, or
          :class:`AcceptInvalidHeader` instance
        * object of any other type that returns a value for ``__str__``

        If `other` is a valid header value or an :class:`AcceptValidHeader`
        instance, then a new :class:`AcceptValidHeader` instance with the valid
        header value is returned.

        If `other` is ``None``, an :class:`AcceptNoHeader` instance, an invalid
        header value, or an :class:`AcceptInvalidHeader` instance, a new
        :class:`AcceptNoHeader` instance is returned.
        """
        if isinstance(other, AcceptValidHeader):
            return AcceptValidHeader(header_value=other.header_value)
        if isinstance(other, (AcceptNoHeader, AcceptInvalidHeader)):
            return AcceptNoHeader()
        return self._add_instance_and_non_accept_type(instance=self, other=other)

    def __radd__(self, other):
        """
        Add to header, creating a new header object.

        See the docstring for :meth:`AcceptValidHeader.__add__`.
        """
        return self._add_instance_and_non_accept_type(instance=self, other=other, instance_on_the_right=True)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def __str__(self):
        """Return the ``str`` ``'<invalid header value>'``."""
        return '<invalid header value>'

    def _add_instance_and_non_accept_type(self, instance, other, instance_on_the_right=False):
        if other is None:
            return AcceptNoHeader()
        other_header_value = self._python_value_to_header_str(value=other)
        try:
            return AcceptValidHeader(header_value=other_header_value)
        except ValueError:
            return AcceptNoHeader()