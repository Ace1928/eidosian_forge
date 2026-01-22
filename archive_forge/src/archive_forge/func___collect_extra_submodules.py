import os
import importlib
import sys
def __collect_extra_submodules(enable_debug_print=False):

    def modules_filter(module):
        return all((not module.startswith('_'), not module.startswith('python-'), os.path.isdir(os.path.join(_extra_submodules_init_path, module))))
    if sys.version_info[0] < 3:
        if enable_debug_print:
            print('Extra submodules is loaded only for Python 3')
        return []
    __INIT_FILE_PATH = os.path.abspath(__file__)
    _extra_submodules_init_path = os.path.dirname(__INIT_FILE_PATH)
    return filter(modules_filter, os.listdir(_extra_submodules_init_path))