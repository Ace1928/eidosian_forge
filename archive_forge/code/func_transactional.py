from inspect import getmembers, isclass, isfunction
from .util import _cfg, getargspec
def transactional(ignore_redirects=True):
    """
    If utilizing the :mod:`pecan.hooks` ``TransactionHook``, allows you
    to flag a controller method or class as being wrapped in a transaction,
    regardless of HTTP method.

    :param ignore_redirects: Indicates if the hook should ignore redirects
                             for this controller or not.
    """

    def deco(f):
        if isclass(f):
            for meth in [m[1] for m in getmembers(f) if isfunction(m[1])]:
                if getattr(meth, 'exposed', False):
                    _cfg(meth)['transactional'] = True
                    _cfg(meth)['transactional_ignore_redirects'] = _cfg(meth).get('transactional_ignore_redirects', ignore_redirects)
        else:
            _cfg(f)['transactional'] = True
            _cfg(f)['transactional_ignore_redirects'] = ignore_redirects
        return f
    return deco