import sys
from functools import partial
from pydev_ipython.version import check_version
class ImportDenier(object):
    """Import Hook that will guard against bad Qt imports
    once IPython commits to a specific binding
    """

    def __init__(self):
        self.__forbidden = set()

    def forbid(self, module_name):
        sys.modules.pop(module_name, None)
        self.__forbidden.add(module_name)

    def find_module(self, fullname, path=None):
        if path:
            return
        if fullname in self.__forbidden:
            return self

    def load_module(self, fullname):
        raise ImportError('\n    Importing %s disabled by IPython, which has\n    already imported an Incompatible QT Binding: %s\n    ' % (fullname, loaded_api()))