import logging 
class PlatformPlugin(Plugin):
    """Platform-level plugin registration"""
    registry = []

    @classmethod
    def match(cls, key):
        """Determine what platform module to load
        
        key -- (sys.platform,os.name) key to load 
        """
        for possible in key:
            for plugin in cls.registry:
                if plugin.name == possible:
                    return plugin
        raise KeyError('No platform plugin registered for %s' % (key,))