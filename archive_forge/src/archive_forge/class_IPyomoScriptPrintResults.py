from pyomo.common.plugin_base import (
class IPyomoScriptPrintResults(Interface):

    def apply(self, **kwds):
        """Apply results printing step in the Pyomo script"""