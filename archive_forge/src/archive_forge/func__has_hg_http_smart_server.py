from ... import version_info  # noqa: F401
from ... import controldir, errors
from ... import transport as _mod_transport
@staticmethod
def _has_hg_http_smart_server(transport, external_url):
    """Check if there is a Mercurial smart server at the remote location.

        :param transport: Transport to check
        :param externa_url: External URL for transport
        :return: Boolean indicating whether transport is backed onto hg
        """
    from breezy.urlutils import urlparse
    parsed_url = urlparse.urlparse(external_url)
    parsed_url = parsed_url._replace(query='cmd=capabilities')
    url = urlparse.urlunparse(parsed_url)
    resp = transport.request('GET', url, headers={'Accept': 'application/mercurial-0.1'})
    if resp.status == 404:
        return False
    if resp.status == 406:
        return False
    ct = resp.getheader('Content-Type')
    if ct is None:
        return False
    return ct.startswith('application/mercurial')