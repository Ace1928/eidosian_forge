import base64
import binascii
import re
import string
import six
def JavaEscape(s):
    """Escapes a string so it can be inserted in a Java string or char literal.

  Follows the Java Language Specification for "Escape Sequences for Character
  and String Literals":

  https://docs.oracle.com/javase/tutorial/java/data/characters.html

  Escapes unprintable and non-ASCII characters.  The resulting string consists
  entirely of ASCII characters.

  This operation is NOT idempotent.

  Sample usage:
    >>> JavaEscape('single\\'double"\\n\\x00')
    'single\\\\\\'double\\\\"\\\\n\\\\000'

  Args:
    s: string to be escaped

  Returns:
    escaped string
  """
    s_esc = _JAVA_ESCAPE_RE.sub(lambda m: _JAVA_ESCAPE_MAP[m.group(0)], s)
    return UnicodeEscape(s_esc)