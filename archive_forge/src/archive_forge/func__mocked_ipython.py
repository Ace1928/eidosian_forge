import sys
from unittest import TestCase
from contextlib import contextmanager
from IPython.display import Markdown, Image
from ipywidgets import widget_output
@contextmanager
def _mocked_ipython(self, get_ipython, clear_output):
    """ Context manager that monkeypatches get_ipython and clear_output """
    original_clear_output = widget_output.clear_output
    original_get_ipython = widget_output.get_ipython
    widget_output.get_ipython = get_ipython
    widget_output.clear_output = clear_output
    try:
        yield
    finally:
        widget_output.clear_output = original_clear_output
        widget_output.get_ipython = original_get_ipython