import atexit
import operator
import os
import sys
import threading
import time
import traceback as _traceback
import warnings
import subprocess
import functools
from more_itertools import always_iterable
@staticmethod
def _get_true_argv():
    """Retrieve all real arguments of the python interpreter.

        ...even those not listed in ``sys.argv``

        :seealso: http://stackoverflow.com/a/28338254/595220
        :seealso: http://stackoverflow.com/a/6683222/595220
        :seealso: http://stackoverflow.com/a/28414807/595220
        """
    try:
        char_p = ctypes.c_wchar_p
        argv = ctypes.POINTER(char_p)()
        argc = ctypes.c_int()
        ctypes.pythonapi.Py_GetArgcArgv(ctypes.byref(argc), ctypes.byref(argv))
        _argv = argv[:argc.value]
        argv_len, is_command, is_module = (len(_argv), False, False)
        try:
            m_ind = _argv.index('-m')
            if m_ind < argv_len - 1 and _argv[m_ind + 1] in ('-c', '-m'):
                "\n                    In some older Python versions `-m`'s argument may be\n                    substituted with `-c`, not `-m`\n                    "
                is_module = True
        except (IndexError, ValueError):
            m_ind = None
        try:
            c_ind = _argv.index('-c')
            if c_ind < argv_len - 1 and _argv[c_ind + 1] == '-c':
                is_command = True
        except (IndexError, ValueError):
            c_ind = None
        if is_module:
            "It's containing `-m -m` sequence of arguments"
            if is_command and c_ind < m_ind:
                "There's `-c -c` before `-m`"
                raise RuntimeError("Cannot reconstruct command from '-c'. Ref: https://github.com/cherrypy/cherrypy/issues/1545")
            original_module = sys.argv[0]
            if not os.access(original_module, os.R_OK):
                "There's no such module exist"
                raise AttributeError("{} doesn't seem to be a module accessible by current user".format(original_module))
            del _argv[m_ind:m_ind + 2]
            _argv.insert(m_ind, original_module)
        elif is_command:
            "It's containing just `-c -c` sequence of arguments"
            raise RuntimeError("Cannot reconstruct command from '-c'. Ref: https://github.com/cherrypy/cherrypy/issues/1545")
    except AttributeError:
        "It looks Py_GetArgcArgv's completely absent in some environments\n\n            It is known, that there's no Py_GetArgcArgv in MS Windows and\n            ``ctypes`` module is completely absent in Google AppEngine\n\n            :seealso: https://github.com/cherrypy/cherrypy/issues/1506\n            :seealso: https://github.com/cherrypy/cherrypy/issues/1512\n            :ref: http://bit.ly/2gK6bXK\n            "
        raise NotImplementedError
    else:
        return _argv