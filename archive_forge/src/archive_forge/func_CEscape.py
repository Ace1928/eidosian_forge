import re
def CEscape(text, as_utf8) -> str:
    """Escape a bytes string for use in an text protocol buffer.

  Args:
    text: A byte string to be escaped.
    as_utf8: Specifies if result may contain non-ASCII characters.
        In Python 3 this allows unescaped non-ASCII Unicode characters.
        In Python 2 the return value will be valid UTF-8 rather than only ASCII.
  Returns:
    Escaped string (str).
  """
    text_is_unicode = isinstance(text, str)
    if as_utf8:
        if text_is_unicode:
            return text.translate(_str_escapes)
        else:
            return _DecodeUtf8EscapeErrors(text)
    else:
        if text_is_unicode:
            text = text.encode('utf-8')
        return ''.join([_byte_escapes[c] for c in text])