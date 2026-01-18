from urllib.parse import urlparse, urlunparse
def web_root_for_service_root(service_root):
    """Turn a service root URL into a web root URL.

    This is done heuristically, not with a lookup.
    """
    service_root = lookup_service_root(service_root)
    web_root_uri = urlparse(service_root)
    web_root_uri = web_root_uri._replace(path='')
    web_root_uri = web_root_uri._replace(netloc=web_root_uri.netloc.replace('api.', '', 1))
    return str(web_root_uri)