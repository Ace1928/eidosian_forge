from collections import namedtuple
import re
import textwrap
import warnings
def accept_charset_property():
    doc = '\n        Property representing the ``Accept-Charset`` header.\n\n        (:rfc:`RFC 7231, section 5.3.3 <7231#section-5.3.3>`)\n\n        The header value in the request environ is parsed and a new object\n        representing the header is created every time we *get* the value of the\n        property. (*set* and *del* change the header value in the request\n        environ, and do not involve parsing.)\n    '
    ENVIRON_KEY = 'HTTP_ACCEPT_CHARSET'

    def fget(request):
        """Get an object representing the header in the request."""
        return create_accept_charset_header(header_value=request.environ.get(ENVIRON_KEY))

    def fset(request, value):
        """
        Set the corresponding key in the request environ.

        `value` can be:

        * ``None``
        * a ``str`` header value
        * a ``dict``, where keys are charsets and values are qvalues
        * a ``tuple`` or ``list``, where each item is a charset ``str`` or a
          ``tuple`` or ``list`` (charset, qvalue) pair (``str``'s and pairs
          can be mixed within the ``tuple`` or ``list``)
        * an :class:`AcceptCharsetValidHeader`, :class:`AcceptCharsetNoHeader`,
          or :class:`AcceptCharsetInvalidHeader` instance
        * object of any other type that returns a value for ``__str__``
        """
        if value is None or isinstance(value, AcceptCharsetNoHeader):
            fdel(request=request)
        else:
            if isinstance(value, (AcceptCharsetValidHeader, AcceptCharsetInvalidHeader)):
                header_value = value.header_value
            else:
                header_value = AcceptCharset._python_value_to_header_str(value=value)
            request.environ[ENVIRON_KEY] = header_value

    def fdel(request):
        """Delete the corresponding key from the request environ."""
        try:
            del request.environ[ENVIRON_KEY]
        except KeyError:
            pass
    return property(fget, fset, fdel, textwrap.dedent(doc))