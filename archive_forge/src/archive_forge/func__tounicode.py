from fontTools.misc.textTools import tostr
def _tounicode(s):
    """Test if a string is valid user input and decode it to unicode string
        using ASCII encoding if it's a bytes string.
        Reject all bytes/unicode input that contains non-XML characters.
        Reject all bytes input that contains non-ASCII characters.
        """
    try:
        s = tostr(s, encoding='ascii', errors='strict')
    except UnicodeDecodeError:
        raise ValueError('Bytes strings can only contain ASCII characters. Use unicode strings for non-ASCII characters.')
    except AttributeError:
        _raise_serialization_error(s)
    if s and _invalid_xml_string.search(s):
        raise ValueError('All strings must be XML compatible: Unicode or ASCII, no NULL bytes or control characters')
    return s