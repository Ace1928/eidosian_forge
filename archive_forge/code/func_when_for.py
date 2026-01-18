from inspect import getmembers, isclass, isfunction
from .util import _cfg, getargspec
def when_for(controller):

    def when(method, **kw):

        def decorate(f):
            _cfg(f)['generic_handler'] = True
            controller._pecan['generic_handlers'][method.upper()] = f
            controller._pecan['allowed_methods'].append(method.upper())
            expose(**kw)(f)
            return f
        return decorate
    return when