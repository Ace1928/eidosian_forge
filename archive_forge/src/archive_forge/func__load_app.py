import argparse
import fnmatch
import importlib
import inspect
import re
import sys
from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils import statemachine
from cliff import app
from cliff import commandmanager
def _load_app(self):
    mod_str, _sep, class_str = self.arguments[0].rpartition('.')
    if not mod_str:
        return
    try:
        importlib.import_module(mod_str)
    except ImportError:
        return
    try:
        cliff_app_class = getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        return
    if not inspect.isclass(cliff_app_class):
        return
    if not issubclass(cliff_app_class, app.App):
        return
    app_arguments = self.options.get('arguments', '').split()
    return cliff_app_class(*app_arguments)