from re import finditer
from xml.sax.saxutils import escape, unescape
def xml_unescape(text):
    """
    This function transforms the "escaped" version suitable
    for well-formed XML formatting into humanly-readable string.

    Note that the default xml.sax.saxutils.unescape() function don't unescape
    some characters that Moses does so we have to manually add them to the
    entities dictionary.

        >>> from xml.sax.saxutils import unescape
        >>> s = ')&#124; &amp; &lt; &gt; &apos; &quot; &#93; &#91;'
        >>> expected = ''')| & < > ' " ] ['''
        >>> xml_unescape(s) == expected
        True

    :param text: The text that needs to be unescaped.
    :type text: str
    :rtype: str
    """
    return unescape(text, entities={'&apos;': "'", '&quot;': '"', '&#124;': '|', '&#91;': '[', '&#93;': ']'})