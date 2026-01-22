def __get_openssl_constructor(name):
    if name in __block_openssl_constructor:
        return __get_builtin_constructor(name)
    try:
        f = getattr(_hashlib, 'openssl_' + name)
        f(usedforsecurity=False)
        return f
    except (AttributeError, ValueError):
        return __get_builtin_constructor(name)