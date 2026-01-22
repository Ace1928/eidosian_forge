from json import loads, dumps
def _missing_token_type(r):
    token = loads(r.text)
    token['token_type'] = 'Bearer'
    r._content = dumps(token).encode()
    return r