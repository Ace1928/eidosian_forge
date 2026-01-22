importers when locating support scripts as well as when importing modules.
import sys
import importlib.machinery # importlib first so we can test #15386 via -m
import importlib.util
import io
import os
def _get_main_module_details(error=ImportError):
    main_name = '__main__'
    saved_main = sys.modules[main_name]
    del sys.modules[main_name]
    try:
        return _get_module_details(main_name)
    except ImportError as exc:
        if main_name in str(exc):
            raise error("can't find %r module in %r" % (main_name, sys.path[0])) from exc
        raise
    finally:
        sys.modules[main_name] = saved_main