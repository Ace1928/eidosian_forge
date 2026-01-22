import importlib
import importlib.util
import inspect
import os
import sys
import types
class DelayedImportErrorModule(types.ModuleType):

    def __init__(self, frame_data, *args, **kwargs):
        self.__frame_data = frame_data
        super().__init__(*args, **kwargs)

    def __getattr__(self, x):
        if x in ('__class__', '__file__', '__frame_data'):
            super().__getattr__(x)
        else:
            fd = self.__frame_data
            raise ModuleNotFoundError(f"No module named '{fd['spec']}'\n\nThis error is lazily reported, having originally occurred in\n  File {fd['filename']}, line {fd['lineno']}, in {fd['function']}\n\n----> {''.join(fd['code_context'] or '').strip()}")