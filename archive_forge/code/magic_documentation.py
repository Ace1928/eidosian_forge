import hashlib
import importlib
from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.core import magic_arguments
import pythran

        Compile and import everything from a Pythran code cell.

        %%pythran
        #pythran export foo(int)
        def foo(x):
            return x + x
        