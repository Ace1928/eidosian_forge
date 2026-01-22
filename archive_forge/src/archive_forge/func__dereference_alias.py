import warnings
from lazr.uri import URI
def _dereference_alias(root, aliases):
    """Dereference what might a URL or an alias for a URL."""
    if root == 'edge':
        warnings.warn("Launchpad edge server no longer exists. Using 'production' instead.", DeprecationWarning)
    if root in aliases:
        return aliases[root]
    scheme, netloc, path, parameters, query, fragment = urlparse(root)
    if scheme != '' and netloc != '':
        return root
    raise ValueError('%s is not a valid URL or an alias for any Launchpad server' % root)