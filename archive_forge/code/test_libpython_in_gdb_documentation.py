import gdb
from Cython.Debugger import libcython
from Cython.Debugger import libpython
from . import test_libcython_in_gdb
from .test_libcython_in_gdb import inferior_python_version

    Test whether types of Python objects are correctly inferred and that
    the right libpython.PySomeTypeObjectPtr classes are instantiated.

    Also test whether values are appropriately formatted (don't be too
    laborious as Lib/test/test_gdb.py already covers this extensively).

    Don't take care of decreffing newly allocated objects as a new
    interpreter is started for every test anyway.
    