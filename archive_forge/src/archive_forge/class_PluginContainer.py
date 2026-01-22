from suds import *
from logging import getLogger
class PluginContainer:
    """
    Plugin container provides easy method invocation.

    @ivar plugins: A list of plugin objects.
    @type plugins: [L{Plugin},]
    @cvar ctxclass: A dict of plugin method / context classes.
    @type ctxclass: dict

    """
    domains = {'init': (InitContext, InitPlugin), 'document': (DocumentContext, DocumentPlugin), 'message': (MessageContext, MessagePlugin)}

    def __init__(self, plugins):
        """
        @param plugins: A list of plugin objects.
        @type plugins: [L{Plugin},]

        """
        self.plugins = plugins

    def __getattr__(self, name):
        domain = self.domains.get(name)
        if not domain:
            raise Exception('plugin domain (%s), invalid' % (name,))
        ctx, pclass = domain
        plugins = [p for p in self.plugins if isinstance(p, pclass)]
        return PluginDomain(ctx, plugins)