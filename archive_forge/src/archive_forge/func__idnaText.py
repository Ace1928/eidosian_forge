def _idnaText(octets: bytes) -> str:
    """
    Convert some IDNA-encoded octets into some human-readable text.

    Currently only used by the tests.

    @param octets: Some bytes representing a hostname.
    @type octets: L{bytes}

    @return: A human-readable domain name.
    @rtype: L{unicode}
    """
    try:
        import idna
    except ImportError:
        return octets.decode('idna')
    else:
        return idna.decode(octets)