import cherrypy
from cherrypy.lib import auth_digest
from cherrypy._cpcompat import ntob
from cherrypy.test import helper
def _test_parametric_digest(self, username, realm):
    test_uri = '/digest/?@/=%2F%40&%f0%9f%99%88=path'
    self.getPage(test_uri)
    assert self.status_code == 401
    msg = 'Digest authentification scheme was not found'
    www_auth_digest = tuple(filter(lambda kv: kv[0].lower() == 'www-authenticate' and kv[1].startswith('Digest '), self.headers))
    assert len(www_auth_digest) == 1, msg
    items = www_auth_digest[0][-1][7:].split(', ')
    tokens = {}
    for item in items:
        key, value = item.split('=')
        tokens[key.lower()] = value
    assert tokens['realm'] == '"localhost"'
    assert tokens['algorithm'] == '"MD5"'
    assert tokens['qop'] == '"auth"'
    assert tokens['charset'] == '"UTF-8"'
    nonce = tokens['nonce'].strip('"')
    base_auth = 'Digest username="%s", realm="%s", nonce="%s", uri="%s", algorithm=MD5, response="%s", qop=auth, nc=%s, cnonce="1522e61005789929"'
    encoded_user = username
    encoded_user = encoded_user.encode('utf-8')
    encoded_user = encoded_user.decode('latin1')
    auth_header = base_auth % (encoded_user, realm, nonce, test_uri, '11111111111111111111111111111111', '00000001')
    auth = auth_digest.HttpDigestAuthorization(auth_header, 'GET')
    ha1 = get_ha1(auth.realm, auth.username)
    response = auth.request_digest(ha1)
    auth_header = base_auth % (encoded_user, realm, nonce, test_uri, response, '00000001')
    self.getPage(test_uri, [('Authorization', auth_header)])