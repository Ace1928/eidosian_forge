from ... import trace
from ...errors import BzrError
from ...hooks import install_lazy_named_hook
from ...config import Option, bool_from_store, option_registry
class QuiltUnapplyError(BzrError):
    _fmt = 'Unable to unapply quilt patches for %(kind)r tree: %(msg)s'

    def __init__(self, kind, msg):
        BzrError.__init__(self)
        self.kind = kind
        if msg is not None and msg.count('\n') == 1:
            msg = msg.strip()
        self.msg = msg