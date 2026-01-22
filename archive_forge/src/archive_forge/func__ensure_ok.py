imported from breezy.bzr.smart.
from io import BytesIO
from .. import config, debug, errors, trace, transport, urlutils
from ..bzr import remote
from ..bzr.smart import client, medium
def _ensure_ok(self, resp):
    if resp[0] != b'ok':
        raise errors.UnexpectedSmartServerResponse(resp)