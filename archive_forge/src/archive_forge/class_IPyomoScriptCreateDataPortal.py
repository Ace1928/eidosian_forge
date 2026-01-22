from pyomo.common.plugin_base import (
class IPyomoScriptCreateDataPortal(Interface):

    def apply(self, **kwds):
        """Apply model data creation step in the Pyomo script"""