def CryptBackendError(handler, config, hash, source='crypt.crypt()'):
    """
    helper to generate standard message when ``crypt.crypt()`` returns invalid result.
    takes care of automatically masking contents of config & hash outside of UTs.
    """
    name = _get_name(handler)
    msg = '%s returned invalid %s hash: config=%s hash=%s' % (source, name, debug_only_repr(config), debug_only_repr(hash))
    raise InternalBackendError(msg)