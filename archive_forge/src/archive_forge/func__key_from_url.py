from requests.auth import AuthBase, HTTPBasicAuth
from requests.compat import urlparse, urlunparse
@staticmethod
def _key_from_url(url):
    parsed = urlparse(url)
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), '', '', '', ''))