import importlib
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import fast_module_type
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.compatibility import all_renames_v2
def _tfmw_import_module(self, name):
    """Lazily loading the modules."""
    if self._tfmw_is_compat_v1 and name != 'app' and (not TFModuleWrapper.compat_v1_usage_recorded):
        TFModuleWrapper.compat_v1_usage_recorded = True
        compat_v1_usage_gauge.get_cell().set(True)
    symbol_loc_info = self._tfmw_public_apis[name]
    if symbol_loc_info[0]:
        module = importlib.import_module(symbol_loc_info[0])
        attr = getattr(module, symbol_loc_info[1])
    else:
        attr = importlib.import_module(symbol_loc_info[1])
    setattr(self._tfmw_wrapped_module, name, attr)
    self.__dict__[name] = attr
    self._fastdict_insert(name, attr)
    return attr