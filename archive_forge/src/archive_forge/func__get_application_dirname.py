import sys
import os
from os import path
from contextlib import contextmanager
def _get_application_dirname(self):
    """
        Return the name of the directory (not a path) that the "main"
        Python script which started this process resides in, or "" if it could
        not be determined or is not appropriate.

        For example, if the script that started the current process was named
        "run.py" in a directory named "foo", and was launched with "python
        run.py", the name "foo" would be returned (this assumes the directory
        name is the name of the app, which seems to be as good of an assumption
        as any).

        """
    dirname = ''
    main_mod = sys.modules.get('__main__', None)
    if main_mod is not None:
        if hasattr(main_mod, '__file__'):
            main_mod_file = path.abspath(main_mod.__file__)
            dirname = path.basename(path.dirname(main_mod_file))
    return dirname