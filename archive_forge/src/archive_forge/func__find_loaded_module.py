import importlib
import logging
import os
import sys
def _find_loaded_module(modpath):
    for k, m in sys.modules.copy().items():
        if k == '__main__':
            continue
        if not hasattr(m, '__file__'):
            continue
        if _likely_same(m.__file__, modpath):
            return m
    return None