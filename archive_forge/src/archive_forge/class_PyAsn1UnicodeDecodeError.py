class PyAsn1UnicodeDecodeError(PyAsn1UnicodeError, UnicodeDecodeError):
    """Unicode text decoding error

    The `PyAsn1UnicodeDecodeError` exception represents a failure to
    deserialize unicode text.

    Apart from inheriting from :class:`PyAsn1UnicodeError`, it also inherits
    from :class:`UnicodeDecodeError` to help the caller catching unicode-related
    errors.
    """