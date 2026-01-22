import nbformat
from traitlets import Enum, default
from .exporter import Exporter
@default('file_extension')
def _file_extension_default(self):
    return '.ipynb'