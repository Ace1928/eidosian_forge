from pyomo.common.plugin_base import (
class IPyomoScriptModifyInstance(Interface):

    def apply(self, **kwds):
        """Modify and return the model instance"""