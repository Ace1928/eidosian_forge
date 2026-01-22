import json
def _null_scope(r):
    token = json.loads(r.text)
    if 'scope' in token and token['scope'] is None:
        token.pop('scope')
    r._content = json.dumps(token).encode()
    return r