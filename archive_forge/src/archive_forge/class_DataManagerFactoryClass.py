import logging
from pyomo.common import Factory
from pyomo.common.plugin_base import PluginError
class DataManagerFactoryClass(Factory):

    def __call__(self, _name=None, args=[], **kwds):
        if _name is None:
            return self
        _name = str(_name)
        if _name in self._cls:
            dm = self._cls[_name](**kwds)
            if not dm.available():
                raise PluginError('Cannot process data in %s files.  The following python packages need to be installed: %s' % (_name, dm.requirements()))
        else:
            dm = UnknownDataManager(type=_name)
        return dm