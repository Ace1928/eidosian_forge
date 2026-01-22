import sys
import fixtures
from functools import wraps
def _import_opts(conf, module, opts, group=None):
    __import__(module)
    conf.register_opts(getattr(sys.modules[module], opts), group=group)