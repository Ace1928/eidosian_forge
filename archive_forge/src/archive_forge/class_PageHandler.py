import string
import sys
import types
import cherrypy
class PageHandler(object):
    """Callable which sets response.body."""

    def __init__(self, callable, *args, **kwargs):
        self.callable = callable
        self.args = args
        self.kwargs = kwargs

    @property
    def args(self):
        """The ordered args should be accessible from post dispatch hooks."""
        return cherrypy.serving.request.args

    @args.setter
    def args(self, args):
        cherrypy.serving.request.args = args
        return cherrypy.serving.request.args

    @property
    def kwargs(self):
        """The named kwargs should be accessible from post dispatch hooks."""
        return cherrypy.serving.request.kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        cherrypy.serving.request.kwargs = kwargs
        return cherrypy.serving.request.kwargs

    def __call__(self):
        try:
            return self.callable(*self.args, **self.kwargs)
        except TypeError:
            x = sys.exc_info()[1]
            try:
                test_callable_spec(self.callable, self.args, self.kwargs)
            except cherrypy.HTTPError:
                raise sys.exc_info()[1]
            except Exception:
                raise x
            raise