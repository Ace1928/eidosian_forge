import os
import importlib
import sys
def __load_extra_py_code_for_module(base, name, enable_debug_print=False):
    module_name = '{}.{}'.format(__name__, name)
    export_module_name = '{}.{}'.format(base, name)
    native_module = sys.modules.pop(module_name, None)
    try:
        py_module = importlib.import_module(module_name)
    except ImportError as err:
        if enable_debug_print:
            print("Can't load Python code for module:", module_name, '. Reason:', err)
        return False
    if not hasattr(base, name):
        setattr(sys.modules[base], name, py_module)
    sys.modules[export_module_name] = py_module
    if native_module:
        setattr(py_module, '_native', native_module)
        for k, v in filter(lambda kv: not hasattr(py_module, kv[0]), native_module.__dict__.items()):
            if enable_debug_print:
                print('    symbol({}): {} = {}'.format(name, k, v))
            setattr(py_module, k, v)
    return True