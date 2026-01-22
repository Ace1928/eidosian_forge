import os
import io
from .._utils import set_module
def _isurl(self, path):
    """Test if path is a net location.  Tests the scheme and netloc."""
    from urllib.parse import urlparse
    scheme, netloc, upath, uparams, uquery, ufrag = urlparse(path)
    return bool(scheme and netloc)