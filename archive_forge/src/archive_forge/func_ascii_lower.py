from __future__ import unicode_literals
import codecs
from .labels import LABELS
def ascii_lower(string):
    """Transform (only) ASCII letters to lower case: A-Z is mapped to a-z.

    :param string: An Unicode string.
    :returns: A new Unicode string.

    This is used for `ASCII case-insensitive
    <http://encoding.spec.whatwg.org/#ascii-case-insensitive>`_
    matching of encoding labels.
    The same matching is also used, among other things,
    for `CSS keywords <http://dev.w3.org/csswg/css-values/#keywords>`_.

    This is different from the :meth:`~py:str.lower` method of Unicode strings
    which also affect non-ASCII characters,
    sometimes mapping them into the ASCII range:

        >>> keyword = u'Bac\\N{KELVIN SIGN}ground'
        >>> assert keyword.lower() == u'background'
        >>> assert ascii_lower(keyword) != keyword.lower()
        >>> assert ascii_lower(keyword) == u'bac\\N{KELVIN SIGN}ground'

    """
    return string.encode('utf8').lower().decode('utf8')