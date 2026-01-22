from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from twisted.internet.defer import Deferred
class _InstanceMethod(NotKnown):

    def __init__(self, im_name, im_self, im_class):
        NotKnown.__init__(self)
        self.my_class = im_class
        self.name = im_name
        im_self.addDependant(self, 0)

    def __call__(self, *args, **kw):
        import traceback
        log.msg(f'instance method {reflect.qual(self.my_class)}.{self.name}')
        log.msg(f'being called with {args!r} {kw!r}')
        traceback.print_stack(file=log.logfile)
        assert 0

    def __setitem__(self, n, obj):
        assert n == 0, 'only zero index allowed'
        if not isinstance(obj, NotKnown):
            method = _constructMethod(self.my_class, self.name, obj)
            self.resolveDependants(method)