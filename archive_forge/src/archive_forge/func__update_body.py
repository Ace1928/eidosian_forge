from urllib import parse
from zunclient.common import base
from zunclient.common import utils
def _update_body(self, host, binary, disabled_reason=None, force_down=None):
    body = {'host': host, 'binary': binary}
    if disabled_reason is not None:
        body['disabled_reason'] = disabled_reason
    if force_down is not None:
        body['forced_down'] = force_down
    return body