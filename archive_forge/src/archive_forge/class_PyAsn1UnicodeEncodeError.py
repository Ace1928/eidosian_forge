class PyAsn1UnicodeEncodeError(PyAsn1UnicodeError, UnicodeEncodeError):
    """Unicode text encoding error

    The `PyAsn1UnicodeEncodeError` exception represents a failure to
    serialize unicode text.

    Apart from inheriting from :class:`PyAsn1UnicodeError`, it also inherits
    from :class:`UnicodeEncodeError` to help the caller catching
    unicode-related errors.
    """