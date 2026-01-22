import os
import sys
from importlib import import_module
import click
from celery.bin.base import CeleryCommand, CeleryOption, handle_preload_options
def _no_ipython(self):
    raise ImportError('no suitable ipython found')