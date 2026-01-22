from traitlets.config.application import Application
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from warnings import warn
from IPython.core.pylabtools import backends
def _show_matplotlib_backend(self, gui, backend):
    """show matplotlib message backend message"""
    if not gui or gui == 'auto':
        print('Using matplotlib backend: %s' % backend)