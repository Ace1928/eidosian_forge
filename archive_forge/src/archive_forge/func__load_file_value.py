import mimetypes
from .widget_core import CoreWidget
from .domwidget import DOMWidget
from .valuewidget import ValueWidget
from .widget import register
from traitlets import Unicode, CUnicode, Bool
from .trait_types import CByteMemoryView
@classmethod
def _load_file_value(cls, filename):
    if getattr(filename, 'read', None) is not None:
        return filename.read()
    else:
        with open(filename, 'rb') as f:
            return f.read()