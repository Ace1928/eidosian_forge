import threading
import warnings
from oslo_utils import importutils
from oslo_utils import timeutils
def warn_eventlet_not_patched(expected_patched_modules=None, what='this library'):
    """Warns if eventlet is being used without patching provided modules.

    :param expected_patched_modules: list of modules to check to ensure that
                                     they are patched (and to warn if they
                                     are not); these names should correspond
                                     to the names passed into the eventlet
                                     monkey_patch() routine. If not provided
                                     then *all* the modules that could be
                                     patched are checked. The currently valid
                                     selection is one or multiple of
                                     ['MySQLdb', '__builtin__', 'all', 'os',
                                     'psycopg', 'select', 'socket', 'thread',
                                     'time'] (where 'all' has an inherent
                                     special meaning).
    :type expected_patched_modules: list/tuple/iterable
    :param what: string to merge into the warnings message to identify
                 what is being checked (used in forming the emitted warnings
                 message).
    :type what: string
    """
    if not expected_patched_modules:
        expanded_patched_modules = _ALL_PATCH.copy()
    else:
        expanded_patched_modules = set()
        for m in expected_patched_modules:
            if m == 'all':
                expanded_patched_modules.update(_ALL_PATCH)
            elif m not in _ALL_PATCH:
                raise ValueError("Unknown module '%s' requested to check if patched" % m)
            else:
                expanded_patched_modules.add(m)
    if EVENTLET_AVAILABLE:
        try:
            maybe_patched = bool(_patcher.already_patched)
        except AttributeError:
            maybe_patched = True
        if maybe_patched:
            not_patched = []
            for m in sorted(expanded_patched_modules):
                if not _patcher.is_monkey_patched(m):
                    not_patched.append(m)
            if not_patched:
                warnings.warn('It is highly recommended that when eventlet is used that the %s modules are monkey patched when using %s (to avoid spurious or unexpected lock-ups and/or hangs)' % (not_patched, what), RuntimeWarning, stacklevel=3)