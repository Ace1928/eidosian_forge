from urllib.parse import urljoin
from twisted.web import resource, server, static, util
def test_site():
    r = resource.Resource()
    r.putChild(b'text', static.Data(b'Works', 'text/plain'))
    r.putChild(b'html', static.Data(b"<body><p class='one'>Works</p><p class='two'>World</p></body>", 'text/html'))
    r.putChild(b'enc-gb18030', static.Data(b'<p>gb18030 encoding</p>', 'text/html; charset=gb18030'))
    r.putChild(b'redirect', util.Redirect(b'/redirected'))
    r.putChild(b'redirect-no-meta-refresh', NoMetaRefreshRedirect(b'/redirected'))
    r.putChild(b'redirected', static.Data(b'Redirected here', 'text/plain'))
    return server.Site(r)