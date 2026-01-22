def __hash_new(name, data=b'', **kwargs):
    """new(name, data=b'') - Return a new hashing object using the named algorithm;
    optionally initialized with data (which must be a bytes-like object).
    """
    if name in __block_openssl_constructor:
        return __get_builtin_constructor(name)(data, **kwargs)
    try:
        return _hashlib.new(name, data, **kwargs)
    except ValueError:
        return __get_builtin_constructor(name)(data)