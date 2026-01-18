import mimetypes
from .widget_core import CoreWidget
from .domwidget import DOMWidget
from .valuewidget import ValueWidget
from .widget import register
from traitlets import Unicode, CUnicode, Bool
from .trait_types import CByteMemoryView

        Convenience method for reading a file into `value`.

        Parameters
        ----------
        filename: str
            The location of a file to read into value from disk.
        