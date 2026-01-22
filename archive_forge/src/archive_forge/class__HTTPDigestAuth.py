import requests
class _HTTPDigestAuth(requests.auth.HTTPDigestAuth):
    init = _ThreadingDescriptor('init', True)
    last_nonce = _ThreadingDescriptor('last_nonce', '')
    nonce_count = _ThreadingDescriptor('nonce_count', 0)
    chal = _ThreadingDescriptor('chal', {})
    pos = _ThreadingDescriptor('pos', None)
    num_401_calls = _ThreadingDescriptor('num_401_calls', 1)