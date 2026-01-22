imported from breezy.bzr.smart.
from io import BytesIO
from .. import config, debug, errors, trace, transport, urlutils
from ..bzr import remote
from ..bzr.smart import client, medium
def _serialise_optional_mode(self, mode):
    if mode is None:
        return b''
    else:
        return ('%d' % mode).encode('ascii')