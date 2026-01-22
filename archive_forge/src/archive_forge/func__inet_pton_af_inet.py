from struct import unpack as _unpack, pack as _pack
def _inet_pton_af_inet(ip_string):
    """
    Convert an IP address in string format (123.45.67.89) to the 32-bit packed
    binary format used in low-level network functions. Differs from inet_aton
    by only support decimal octets. Using octal or hexadecimal values will
    raise a ValueError exception.
    """
    if isinstance(ip_string, str):
        invalid_addr = OSError('illegal IP address string %r' % ip_string)
        tokens = ip_string.split('.')
        if len(tokens) == 4:
            words = []
            for token in tokens:
                if token.startswith('0x') or (token.startswith('0') and len(token) > 1):
                    raise invalid_addr
                try:
                    octet = int(token)
                except ValueError:
                    raise invalid_addr
                if octet >> 8 != 0:
                    raise invalid_addr
                words.append(_pack('B', octet))
            return b''.join(words)
        else:
            raise invalid_addr
    raise TypeError(f'inet_pton() argument 2 must be str, not {type(ip_string)}')