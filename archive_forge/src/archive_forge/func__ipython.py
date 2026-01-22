import os
import sys
from importlib import import_module
import click
from celery.bin.base import CeleryCommand, CeleryOption, handle_preload_options
def _ipython(locals):
    from IPython import start_ipython
    start_ipython(argv=[], user_ns=locals)