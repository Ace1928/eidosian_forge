from pyomo.common.plugin_base import (
class IPyomoScriptPrintModel(Interface):

    def apply(self, **kwds):
        """Apply model printing step in the Pyomo script"""