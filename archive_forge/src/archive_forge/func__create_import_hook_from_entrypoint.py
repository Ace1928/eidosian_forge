import importlib  # noqa: F401
import sys
import threading
def _create_import_hook_from_entrypoint(entrypoint):

    def import_hook(module):
        __import__(entrypoint.module_name)
        callback = sys.modules[entrypoint.module_name]
        for attr in entrypoint.attrs:
            callback = getattr(callback, attr)
        return callback(module)
    return import_hook