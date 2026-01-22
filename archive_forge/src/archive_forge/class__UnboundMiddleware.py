from webob.compat import (
from webob.request import Request
from webob.exc import HTTPException
class _UnboundMiddleware(object):
    """A `wsgify.middleware` invocation that has not yet wrapped a
    middleware function; the intermediate object when you do
    something like ``@wsgify.middleware(RequestClass=Foo)``
    """

    def __init__(self, wrapper_class, app, kw):
        self.wrapper_class = wrapper_class
        self.app = app
        self.kw = kw

    def __repr__(self):
        return '<%s at %s wrapping %r>' % (self.__class__.__name__, id(self), self.app)

    def __call__(self, func, app=None):
        if app is None:
            app = self.app
        return self.wrapper_class.middleware(func, app=app, **self.kw)