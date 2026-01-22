from pyomo.common.plugin_base import (
class IPyomoScriptPreprocess(Interface):

    def apply(self, **kwds):
        """Apply preprocessing step in the Pyomo script"""