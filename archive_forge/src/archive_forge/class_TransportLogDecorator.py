import time
import types
from ..trace import mutter
from ..transport import decorator
class TransportLogDecorator(decorator.TransportDecorator):
    """Decorator for Transports that logs interesting operations to brz.log.

    In general we want to log things that usually take a network round trip
    and may be slow.

    Not all operations are logged yet.

    See also TransportTraceDecorator, that records a machine-readable log in
    memory for eg testing.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        def _make_hook(hookname):

            def _hook(relpath, *args, **kw):
                return self._log_and_call(hookname, relpath, *args, **kw)
            return _hook
        for methodname in ('append_bytes', 'append_file', 'copy_to', 'delete', 'get', 'has', 'open_write_stream', 'mkdir', 'move', 'put_bytes', 'put_bytes_non_atomic', 'put_file put_file_non_atomic', 'list_dir', 'lock_read', 'lock_write', 'readv', 'rename', 'rmdir', 'stat', 'ulock'):
            setattr(self, methodname, _make_hook(methodname))

    @classmethod
    def _get_url_prefix(self):
        return 'log+'

    def iter_files_recursive(self):
        mutter('%s %s' % ('iter_files_recursive', self._decorated.base))
        return self._call_and_log_result('iter_files_recursive', (), {})

    def _log_and_call(self, methodname, relpath, *args, **kwargs):
        if kwargs:
            kwargs_str = dict(kwargs)
        else:
            kwargs_str = ''
        mutter('%s %s %s %s' % (methodname, relpath, self._shorten(self._strip_tuple_parens(args)), kwargs_str))
        return self._call_and_log_result(methodname, (relpath,) + args, kwargs)

    def _call_and_log_result(self, methodname, args, kwargs):
        before = time.time()
        try:
            result = getattr(self._decorated, methodname)(*args, **kwargs)
        except Exception as e:
            mutter('  --> %s' % e)
            mutter('      %.03fs' % (time.time() - before))
            raise
        return self._show_result(before, methodname, result)

    def _show_result(self, before, methodname, result):
        result_len = None
        if isinstance(result, types.GeneratorType):
            result = list(result)
            return_result = iter(result)
        else:
            return_result = result
        getvalue = getattr(result, 'getvalue', None)
        if getvalue is not None:
            val = repr(getvalue())
            result_len = len(val)
            shown_result = '%s(%s) (%d bytes)' % (result.__class__.__name__, self._shorten(val), result_len)
        elif methodname == 'readv':
            num_hunks = len(result)
            total_bytes = sum((len(d) for o, d in result))
            shown_result = 'readv response, %d hunks, %d total bytes' % (num_hunks, total_bytes)
            result_len = total_bytes
        else:
            shown_result = self._shorten(self._strip_tuple_parens(result))
        mutter('  --> %s' % shown_result)
        if False:
            elapsed = time.time() - before
            if result_len and elapsed > 0:
                mutter('      %9.03fs %8dkB/s' % (elapsed, result_len / elapsed / 1000))
            else:
                mutter('      %9.03fs' % elapsed)
        return return_result

    def _shorten(self, x):
        if len(x) > 70:
            x = x[:67] + '...'
        return x

    def _strip_tuple_parens(self, t):
        t = repr(t)
        if t[0] == '(' and t[-1] == ')':
            t = t[1:-1]
        return t