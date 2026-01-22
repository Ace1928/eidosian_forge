import sys as _sys
import io
import cherrypy as _cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cperror
from cherrypy.lib import httputil
from cherrypy.lib import is_closable_iterator
class _TrappedResponse(object):
    response = iter([])

    def __init__(self, nextapp, environ, start_response, throws):
        self.nextapp = nextapp
        self.environ = environ
        self.start_response = start_response
        self.throws = throws
        self.started_response = False
        self.response = self.trap(self.nextapp, self.environ, self.start_response)
        self.iter_response = iter(self.response)

    def __iter__(self):
        self.started_response = True
        return self

    def __next__(self):
        return self.trap(next, self.iter_response)

    def close(self):
        if hasattr(self.response, 'close'):
            self.response.close()

    def trap(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except self.throws:
            raise
        except StopIteration:
            raise
        except Exception:
            tb = _cperror.format_exc()
            _cherrypy.log(tb, severity=40)
            if not _cherrypy.request.show_tracebacks:
                tb = ''
            s, h, b = _cperror.bare_error(tb)
            if True:
                s = s.decode('ISO-8859-1')
                h = [(k.decode('ISO-8859-1'), v.decode('ISO-8859-1')) for k, v in h]
            if self.started_response:
                self.iter_response = iter([])
            else:
                self.iter_response = iter(b)
            try:
                self.start_response(s, h, _sys.exc_info())
            except Exception:
                _cherrypy.log(traceback=True, severity=40)
                raise
            if self.started_response:
                return b''.join(b)
            else:
                return b