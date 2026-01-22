import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def _unregister_urlparse_netloc_protocol(protocol):
    """Remove protocol from urlparse netloc parsing.

    Except for tests, you should never use that function. Using it with 'http',
    for example, will break all http transports.
    """
    if protocol in urlutils.urlparse.uses_netloc:
        urlutils.urlparse.uses_netloc.remove(protocol)