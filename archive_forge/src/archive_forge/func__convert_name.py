import sys
import argparse
import os
import warnings
from . import loader, runner
from .signals import installHandler
def _convert_name(name):
    if os.path.isfile(name) and name.lower().endswith('.py'):
        if os.path.isabs(name):
            rel_path = os.path.relpath(name, os.getcwd())
            if os.path.isabs(rel_path) or rel_path.startswith(os.pardir):
                return name
            name = rel_path
        return os.path.normpath(name)[:-3].replace('\\', '.').replace('/', '.')
    return name