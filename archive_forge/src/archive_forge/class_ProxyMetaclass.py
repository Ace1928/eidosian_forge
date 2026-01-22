from .misc import Literal, moduleMember
class ProxyMetaclass(type):
    """ ProxyMetaclass is the meta-class for proxies. """

    def __init__(*args):
        """ Initialise the meta-class. """
        type.__init__(*args)
        proxy = args[0]
        for sub_proxy in proxy.__dict__.values():
            if type(sub_proxy) is ProxyMetaclass:
                sub_proxy.module = proxy.__name__
                for sub_sub_proxy in sub_proxy.__dict__.values():
                    if type(sub_sub_proxy) is ProxyMetaclass:
                        sub_sub_proxy.module = '%s.%s' % (proxy.__name__, sub_sub_proxy.module)
        if not hasattr(proxy, 'module'):
            proxy.module = ''

    def __getattribute__(cls, name):
        try:
            return type.__getattribute__(cls, name)
        except AttributeError:
            if name == 'module':
                raise
            from .qtproxies import LiteralProxyClass
            return type(name, (LiteralProxyClass,), {'module': moduleMember(type.__getattribute__(cls, 'module'), type.__getattribute__(cls, '__name__'))})

    def __str__(cls):
        return moduleMember(type.__getattribute__(cls, 'module'), type.__getattribute__(cls, '__name__'))

    def __or__(self, r_op):
        return Literal('%s|%s' % (self, r_op))

    def __eq__(self, other):
        return str(self) == str(other)