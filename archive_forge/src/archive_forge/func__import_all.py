import hashlib
import importlib
from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.core import magic_arguments
import pythran
def _import_all(self, module):
    """ Import only globals modules. """
    self.shell.push({k: v for k, v in module.__dict__.items() if not k.startswith('__')})