import threading
from _thread import get_ident
from ... import branch as _mod_branch
from ... import debug, errors, osutils, registry, revision, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...lazy_import import lazy_import
from breezy.bzr import bzrdir
from breezy.bzr.bundle import serializer
import tempfile
def _pre_open_hook(transport):
    allowed_transports = getattr(jail_info, 'transports', None)
    if allowed_transports is None:
        return
    abspath = transport.base
    for allowed_transport in allowed_transports:
        try:
            allowed_transport.relpath(abspath)
        except errors.PathNotChild:
            continue
        else:
            return
    raise errors.JailBreak(abspath)